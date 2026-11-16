# realtime_overlay.py - COMPLETE FIXED VERSION
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple, Optional
import json
import base64
from io import BytesIO
import os
import logging
from datetime import datetime

class RealtimeOverlay:
    """Real-time text overlay and editing system - FIXED VERSION"""
    
    def __init__(self):
        self.active_overlays = {}
        self.font_cache = {}
        self.logger = logging.getLogger(__name__)
        
    def create_text_overlay(self, 
                          image: np.ndarray,
                          text_bboxes: List[List[float]],
                          translated_texts: List[str],
                          styles: List[Dict] = None) -> np.ndarray:
        # ✅ MANGA READING ORDER SORTING: TOP FIRST, THEN PROPER HORIZONTAL ORDER
        # Combine all elements and sort by vertical position
        elements = list(zip(text_bboxes, translated_texts, styles or [self._get_default_style()]*len(text_bboxes)))
        
        # PRIMARY SORT: TOP OF PAGE FIRST (smallest Y coordinate = highest on page)
        elements.sort(key=lambda e: (e[0][1], -e[0][0]))  # Y first, then X right-to-left for manga
        
        # Unpack back into sorted lists
        text_bboxes = [e[0] for e in elements]
        translated_texts = [e[1] for e in elements]
        styles = [e[2] for e in elements]
        """
        Create overlay with translated text
        
        Args:
            image: Base image (BGR format from OpenCV)
            text_bboxes: Original text bounding boxes
            translated_texts: Translated texts
            styles: List of style dictionaries for each text
        """
        try:
            # Convert BGR to RGB for PIL
            if len(image.shape) == 3 and image.shape[2] == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)
                
            draw = ImageDraw.Draw(pil_image, 'RGBA')
            
            height, width = image.shape[:2]
            
            for i, (bbox, text) in enumerate(zip(text_bboxes, translated_texts)):
                if not text or not text.strip():
                    continue
                    
                # Get style for this text
                style = styles[i] if styles and i < len(styles) else self._get_default_style()
                
                # Convert normalized coords to pixels
                if all(0 <= v <= 1 for v in bbox):
                    x1 = int(bbox[0] * width)
                    y1 = int(bbox[1] * height)
                    x2 = int(bbox[2] * width)
                    y2 = int(bbox[3] * height)
                else:
                    x1, y1, x2, y2 = [int(coord) for coord in bbox]
                
                # Ensure valid coordinates
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Render text with style
                self._render_text_with_style(draw, text, (x1, y1, x2, y2), style, image.shape)
            
            # Convert back to OpenCV format (BGR)
            result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            return result
            
        except Exception as e:
            self.logger.error(f"Overlay creation failed: {e}")
            # Return original image on error
            return image
    
    def _get_default_style(self) -> Dict:
        """Get default text style for manga"""
        return {
            'font_family': 'Arial',
            'font_size': 14,
            'font_color': (0, 0, 0),  # Black - RGB
            'background_color': (255, 255, 255),  # White - RGB
            'border_color': (0, 0, 0),  # Black - RGB
            'border_width': 1,
            'padding': 8,
            'alignment': 'center'
        }
    
    def _render_text_with_style(self, 
                              draw: ImageDraw.Draw,
                              text: str,
                              bbox: Tuple[int, int, int, int],
                              style: Dict,
                              image_shape: Tuple[int, int]):
        """Render text with full styling - FIXED COLOR HANDLING"""
        try:
            x1, y1, x2, y2 = bbox
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            
            if bbox_width <= 0 or bbox_height <= 0:
                return
            
            # Get font with better error handling
            font = self._get_font(style['font_family'], style['font_size'])
            
            # Wrap text to fit bbox
            lines = self._wrap_text(text, font, bbox_width - style['padding'] * 2)
            if not lines:
                return
            
            # Calculate text height
            try:
                # Modern PIL method
                bbox_list = [font.getbbox(line) for line in lines]
                line_height = max(bbox[3] - bbox[1] for bbox in bbox_list) if bbox_list else font.size
            except AttributeError:
                # Fallback for older PIL
                line_height = font.getsize("A")[1]
            
            total_height = len(lines) * line_height
            
            # ✅ SMART LANGUAGE AWARE FIT ALGORITHM - KEEP TEXT AS LARGE AS POSSIBLE
            max_allowed_height = bbox_height - (style['padding'] * 2)
            max_allowed_width = bbox_width - (style['padding'] * 2)
            
            # Detect language for optimized scaling strategy
            is_japanese = any('\u3040' <= c <= '\u30ff' or '\u4e00' <= c <= '\u9fff' for c in text)
            is_chinese = any('\u4e00' <= c <= '\u9fff' and not '\u3040' <= c <= '\u30ff' for c in text)
            is_korean = any('\uac00' <= c <= '\ud7af' for c in text)
            
            # Language specific scaling profiles
            if is_japanese:
                # Japanese: Tighten line spacing FIRST, font size LAST
                max_line_spacing_reduction = 0.32  # 32% tighter lines
                max_padding_reduction = 0.6
                min_font_size = 9
            elif is_korean:
                # Korean: Adjust line spacing only
                max_line_spacing_reduction = 0.25
                max_padding_reduction = 0.5
                min_font_size = 10
            elif is_chinese:
                # Chinese: Adjust letter spacing first
                max_line_spacing_reduction = 0.28
                max_padding_reduction = 0.55
                min_font_size = 9
            else:
                # Western languages: Reduce everything gradually
                max_line_spacing_reduction = 0.22
                max_padding_reduction = 0.4
                min_font_size = 10
            
            # Initial values
            final_font = font
            final_lines = lines
            final_line_height = line_height
            final_total_height = total_height
            line_spacing_multiplier = 1.0
            padding_multiplier = 1.0
            
            # ✅ PHASE 1: ADJUST LINE SPACING FIRST (no font size reduction yet)
            if final_total_height > max_allowed_height:
                line_spacing_multiplier = min(1.0, max_allowed_height / final_total_height)
                line_spacing_multiplier = max(1.0 - max_line_spacing_reduction, line_spacing_multiplier)
                final_line_height = int(final_line_height * line_spacing_multiplier)
                final_total_height = len(final_lines) * final_line_height
            
            # ✅ PHASE 2: REDUCE PADDING IF NEEDED
            if final_total_height > max_allowed_height:
                padding_reduction = min(max_padding_reduction, (final_total_height - max_allowed_height) / final_total_height + 0.1)
                padding_multiplier = 1.0 - padding_reduction
                adjusted_padding = int(style['padding'] * padding_multiplier)
                max_allowed_height = bbox_height - (adjusted_padding * 2)
                max_allowed_width = bbox_width - (adjusted_padding * 2)
            
            # ✅ PHASE 3: ONLY NOW REDUCE FONT SIZE AS ABSOLUTE LAST RESORT
            target_font_size = style['font_size']
            for attempt in range(12):
                if final_total_height <= max_allowed_height:
                    break
                    
                target_font_size -= 1
                if target_font_size < min_font_size:
                    break
                    
                final_font = self._get_font(style['font_family'], target_font_size)
                final_lines = self._wrap_text(text, final_font, max_allowed_width)
                
                try:
                    bbox_list = [final_font.getbbox(line) for line in final_lines]
                    final_line_height = max(bbox[3] - bbox[1] for bbox in bbox_list) if bbox_list else final_font.size
                except AttributeError:
                    final_line_height = final_font.getsize("A")[1]
                
                # Apply line spacing adjustment
                final_line_height = int(final_line_height * line_spacing_multiplier)
                final_total_height = len(final_lines) * final_line_height
            
            # Use the perfectly fitted values
            font = final_font
            lines = final_lines
            line_height = final_line_height
            total_height = final_total_height
            
            # Draw background if specified
            if style.get('background_color'):
                bg_x1 = x1 - style['padding']
                bg_y1 = y1 + (bbox_height - total_height) // 2 - style['padding']
                bg_x2 = x2 + style['padding']
                bg_y2 = bg_y1 + total_height + style['padding'] * 2
                
                # Ensure background stays within image bounds
                bg_x1 = max(0, bg_x1)
                bg_y1 = max(0, bg_y1)
                bg_x2 = min(image_shape[1], bg_x2)
                bg_y2 = min(image_shape[0], bg_y2)
                
                if bg_x2 > bg_x1 and bg_y2 > bg_y1:
                    # FIX: Handle color format conversion
                    bg_color = self._ensure_rgb_color(style['background_color'])
                    outline_color = self._ensure_rgb_color(style.get('border_color'))
                    
                    draw.rectangle(
                        [bg_x1, bg_y1, bg_x2, bg_y2],
                        fill=bg_color,
                        outline=outline_color,
                        width=style.get('border_width', 0)
                    )
            
            # Draw each line of text
            start_y = y1 + (bbox_height - total_height) // 2
            
            for i, line in enumerate(lines):
                if not line.strip():
                    continue
                    
                # Get text dimensions
                try:
                    # Modern PIL
                    text_bbox = font.getbbox(line)
                    text_width = text_bbox[2] - text_bbox[0]
                except AttributeError:
                    # Older PIL
                    text_width, text_height = font.getsize(line)
                
                # Calculate text position based on alignment
                if style['alignment'] == 'center':
                    text_x = x1 + (bbox_width - text_width) // 2
                elif style['alignment'] == 'right':
                    text_x = x2 - text_width - style['padding']
                else:  # left
                    text_x = x1 + style['padding']
                
                text_y = start_y + i * line_height
                
                # Ensure text stays within bounds
                text_x = max(0, text_x)
                text_y = max(0, text_y)
                
                # FIX: Handle font color format
                font_color = self._ensure_rgb_color(style['font_color'])
                
                # Draw text with optional outline
                if style.get('border_width', 0) > 0:
                    # Draw outline
                    outline_color = self._ensure_rgb_color(style['border_color'])
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            draw.text(
                                (text_x + dx, text_y + dy), 
                                line, 
                                font=font, 
                                fill=outline_color
                            )
                
                # Draw main text
                draw.text(
                    (text_x, text_y), 
                    line, 
                    font=font, 
                    fill=font_color
                )
                
        except Exception as e:
            self.logger.warning(f"Text rendering failed: {e}")
    
    def _ensure_rgb_color(self, color) -> Tuple[int, int, int]:
        """Ensure color is in RGB format"""
        if not color:
            return (0, 0, 0)
        
        if isinstance(color, list):
            color = tuple(color)
            
        if len(color) == 4:  # RGBA
            return color[:3]  # Convert to RGB
        elif len(color) == 3:  # RGB
            return color
        else:
            self.logger.warning(f"Invalid color format: {color}, using black")
            return (0, 0, 0)
    
    def _get_font(self, font_family: str, font_size: int) -> ImageFont.FreeTypeFont:
        """Get font from cache or load it with FULL UNICODE CJK SUPPORT"""
        cache_key = f"{font_family}_{font_size}"
        if cache_key not in self.font_cache:
            try:
                # Try to load the specified font
                font = ImageFont.truetype(font_family, font_size)
            except (OSError, IOError):
                try:
                    # ✅ UNIVERSAL FONTS WITH FULL CJK / ALL LANGUAGE SUPPORT FIRST
                    fallback_fonts = [
                        # Windows system fonts that support ALL languages
                        'C:/Windows/Fonts/msgothic.ttc',    # Japanese + Full CJK
                        'C:/Windows/Fonts/mingliu.ttc',     # Chinese Traditional
                        'C:/Windows/Fonts/msyh.ttc',        # Chinese Simplified
                        'C:/Windows/Fonts/malgun.ttf',      # Korean
                        'C:/Windows/Fonts/arialuni.ttf',    # Arial Unicode MS - FULL UNICODE
                        'C:/Windows/Fonts/segoeui.ttf',     # Segoe UI - modern unicode support
                        'C:/Windows/Fonts/arial.ttf', 
                        
                        # Linux cross-language fonts
                        'NotoSansCJK-Regular.ttc',
                        'NotoSans-Regular.ttf',
                        'DejaVuSans.ttf', 
                        'LiberationSans-Regular.ttf',
                        '/usr/share/fonts/truetype/freefont/FreeSans.ttf'
                    ]
                    for fallback in fallback_fonts:
                        try:
                            if os.path.exists(fallback):
                                font = ImageFont.truetype(fallback, font_size)
                                break
                        except (OSError, IOError):
                            continue
                    else:
                        # All fallbacks failed, use default
                        font = ImageFont.load_default()
                        self.logger.warning(f"Using default font, requested font not found: {font_family}")
                except Exception:
                    font = ImageFont.load_default()
            self.font_cache[cache_key] = font
        return self.font_cache[cache_key]
    
    def _wrap_text(self, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
        """Wrap text to fit within max width - INTELLIGENT NO WORD BREAK"""
        if max_width <= 0:
            return [text]
            
        # Handle CJK languages properly (no spaces between characters)
        is_cjk = any('\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff' or '\uac00' <= c <= '\ud7af' for c in text)
        
        lines = []
        current_line = []
        
        if is_cjk:
            # CJK intelligent character by character wrapping
            current_length = 0
            current_segment = []
            
            for char in text:
                try:
                    bbox = font.getbbox(char)
                    char_width = bbox[2] - bbox[0]
                except AttributeError:
                    char_width, _ = font.getsize(char)
                
                if current_length + char_width > max_width and current_segment:
                    lines.append(''.join(current_segment))
                    current_segment = []
                    current_length = 0
                
                current_segment.append(char)
                current_length += char_width
            
            if current_segment:
                lines.append(''.join(current_segment))
            
            return lines
        
        # Western language word preserving wrapping
        words = text.split()
        if not words:
            return []
        
        for word in words:
            # Test if adding this word exceeds max width
            test_line = ' '.join(current_line + [word])
            try:
                bbox = font.getbbox(test_line)
                test_width = bbox[2] - bbox[0]
            except AttributeError:
                test_width, _ = font.getsize(test_line)
            
            if test_width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                
                # Check if single word is too long
                try:
                    word_bbox = font.getbbox(word)
                    word_width = word_bbox[2] - word_bbox[0]
                except AttributeError:
                    word_width, _ = font.getsize(word)
                
                if word_width > max_width:
                    # Split very long words only when absolutely necessary
                    lines.extend(self._split_long_word(word, font, max_width))
                else:
                    current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines
    
    def _split_long_word(self, word: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
        """Split a long word that doesn't fit in the width"""
        if not word:
            return []
            
        parts = []
        current_part = ""
        
        for char in word:
            test_part = current_part + char
            try:
                bbox = font.getbbox(test_part)
                test_width = bbox[2] - bbox[0]
            except AttributeError:
                test_width, _ = font.getsize(test_part)
                
            if test_width <= max_width:
                current_part = test_part
            else:
                if current_part:
                    parts.append(current_part)
                current_part = char
                
        if current_part:
            parts.append(current_part)
            
        return parts
    
    def create_editable_overlay_data(self,
                                   image: np.ndarray,
                                   text_elements: List[Dict]) -> Dict:
        """
        Create data structure for frontend editable overlay
        """
        overlay_data = {
            'image_width': image.shape[1],
            'image_height': image.shape[0],
            'text_elements': [],
            'version': '1.0',
            'timestamp': self._get_timestamp()
        }
        
        for i, element in enumerate(text_elements):
            text_element = {
                'id': f"text_{i}",
                'original_text': element.get('original_text', ''),
                'translated_text': element.get('translated_text', ''),
                'bbox': element['bbox'],
                'style': element.get('style', self._get_default_style()),
                'editable': element.get('editable', True),
                'movable': element.get('movable', True)
            }
            overlay_data['text_elements'].append(text_element)
        
        return overlay_data
    
    def apply_editor_changes(self,
                           image: np.ndarray,
                           overlay_data: Dict) -> np.ndarray:
        """Apply changes from frontend editor"""
        try:
            text_bboxes = []
            translated_texts = []
            styles = []
            
            for element in overlay_data.get('text_elements', []):
                text_bboxes.append(element['bbox'])
                translated_texts.append(element['translated_text'])
                styles.append(element['style'])
            
            return self.create_text_overlay(image, text_bboxes, translated_texts, styles)
            
        except Exception as e:
            self.logger.error(f"Applying editor changes failed: {e}")
            return image
    
    def image_to_base64(self, image: np.ndarray) -> str:
        """Convert numpy image to base64 string"""
        try:
            success, buffer = cv2.imencode('.png', image)
            if not success:
                raise ValueError("Could not encode image to PNG")
            
            img_str = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/png;base64,{img_str}"
        except Exception as e:
            self.logger.error(f"Image to base64 conversion failed: {e}")
            return ""
    
    def base64_to_image(self, base64_str: str) -> np.ndarray:
        """Convert base64 string to numpy image"""
        try:
            if ',' in base64_str:
                base64_str = base64_str.split(',')[1]
            
            img_data = base64.b64decode(base64_str)
            img_array = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode image from base64")
            return img
        except Exception as e:
            self.logger.error(f"Base64 to image conversion failed: {e}")
            # Return a blank image as fallback
            return np.ones((100, 100, 3), dtype=np.uint8) * 255
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for versioning"""
        return datetime.now().isoformat()
    
    def validate_overlay_data(self, overlay_data: Dict) -> bool:
        """Validate overlay data structure"""
        required_keys = ['image_width', 'image_height', 'text_elements']
        if not all(key in overlay_data for key in required_keys):
            return False
        
        for element in overlay_data.get('text_elements', []):
            if 'bbox' not in element or 'translated_text' not in element:
                return False
        
        return True

    def get_supported_fonts(self) -> List[str]:
        """Get list of supported fonts"""
        return ['Arial', 'Helvetica', 'Times New Roman', 'Courier New']
    
    def clear_font_cache(self):
        """Clear the font cache"""
        self.font_cache.clear()
        self.logger.info("Font cache cleared")
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        return {
            'font_cache_size': len(self.font_cache),
            'active_overlays': len(self.active_overlays),
            'timestamp': self._get_timestamp()
        }

# Global instance
realtime_overlay = RealtimeOverlay()