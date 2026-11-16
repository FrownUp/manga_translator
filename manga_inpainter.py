# manga_inpainter.py
import cv2
import numpy as np
import torch
from PIL import Image
import base64
from io import BytesIO
from typing import List, Tuple, Optional, Dict, Any
import logging

class MangaInpainter:
    """
    Inpainting system for manga images with multiple backends
    """
    
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.logger = logging.getLogger(__name__)
        
    def load_models(self):
        """Load available inpainting models"""
        try:
            # Try to load LaMa inpainting
            from lama_cleaner.model_manager import ModelManager
            from lama_cleaner.schema import Config
            
            self.models["lama"] = ModelManager(name="lama", device=self.device)
            self.logger.info("LaMa inpainting model loaded")
        except ImportError as e:
            self.logger.warning(f"LaMa not available: {e}")
        
        try:
            # Try to load Stable Diffusion inpainting
            from diffusers import StableDiffusionInpaintPipeline
            
            self.models["sd"] = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)
            self.logger.info("Stable Diffusion inpainting model loaded")
        except ImportError as e:
            self.logger.warning(f"Stable Diffusion not available: {e}")
            
        try:
            # OpenCV-based inpainting (fallback)
            self.models["opencv"] = "available"
            self.logger.info("OpenCV inpainting available")
        except Exception as e:
            self.logger.warning(f"OpenCV inpainting not available: {e}")
    
    def create_mask_from_bboxes(self, 
                              image: np.ndarray, 
                              bboxes: List[List[float]], 
                              expand_pixels: int = 5) -> np.ndarray:
        """
        Create mask from bounding boxes for inpainting
        
        Args:
            image: Input image as numpy array
            bboxes: List of bounding boxes in format [x1, y1, x2, y2]
            expand_pixels: Number of pixels to expand the mask around each bbox
            
        Returns:
            Binary mask where regions to inpaint are white (255)
        """
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        for bbox in bboxes:
            if all(0 <= v <= 1 for v in bbox):
                # Normalized coordinates
                x1 = int(bbox[0] * width)
                y1 = int(bbox[1] * height)
                x2 = int(bbox[2] * width)
                y2 = int(bbox[3] * height)
            else:
                # Pixel coordinates
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Expand the region slightly
            x1 = max(0, x1 - expand_pixels)
            y1 = max(0, y1 - expand_pixels)
            x2 = min(width, x2 + expand_pixels)
            y2 = min(height, y2 + expand_pixels)
            
            # Fill the rectangle in the mask
            mask[y1:y2, x1:x2] = 255
        
        return mask
    
    def create_text_removal_mask(self, 
                               image: np.ndarray,
                               text_bboxes: List[List[float]],
                               character_bboxes: List[List[float]] = None,
                               exclude_characters: bool = True) -> np.ndarray:
        """
        Create mask for text removal while optionally preserving characters
        
        Args:
            image: Input image
            text_bboxes: Bounding boxes of text regions to remove
            character_bboxes: Bounding boxes of characters to preserve
            exclude_characters: Whether to exclude character regions from masking
            
        Returns:
            Binary mask for inpainting
        """
        mask = self.create_mask_from_bboxes(image, text_bboxes)
        
        if exclude_characters and character_bboxes:
            height, width = image.shape[:2]
            character_mask = np.zeros((height, width), dtype=np.uint8)
            
            for bbox in character_bboxes:
                if all(0 <= v <= 1 for v in bbox):
                    x1 = int(bbox[0] * width)
                    y1 = int(bbox[1] * height)
                    x2 = int(bbox[2] * width)
                    y2 = int(bbox[3] * height)
                else:
                    x1, y1, x2, y2 = [int(coord) for coord in bbox]
                
                character_mask[y1:y2, x1:x2] = 255
            
            # Remove character regions from the text mask
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(character_mask))
        
        return mask
    
    def inpaint_with_opencv(self, 
                          image: np.ndarray, 
                          mask: np.ndarray, 
                          method: str = "telea") -> np.ndarray:
        """
        Inpainting using OpenCV methods
        
        Args:
            image: Input image (BGR format)
            mask: Binary mask (white = regions to inpaint)
            method: "telea" or "ns" for different algorithms
            
        Returns:
            Inpainted image
        """
        if method == "telea":
            inpainted = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        else:  # "ns"
            inpainted = cv2.inpaint(image, mask, 3, cv2.INPAINT_NS)
        
        return inpainted
    
    def inpaint_with_lama(self, 
                         image: np.ndarray, 
                         mask: np.ndarray,
                         config: Dict[str, Any] = None) -> np.ndarray:
        """
        Inpainting using LaMa model
        """
        if "lama" not in self.models:
            raise ValueError("LaMa model not loaded")
        
        from lama_cleaner.schema import Config as LamaConfig
        
        default_config = {
            "ldm_steps": 20,
            "ldm_sampler": "plms",
            "hd_strategy": "Crop",
            "hd_strategy_crop_margin": 128,
            "hd_strategy_crop_trigger_size": 512,
            "hd_strategy_resize_limit": 1024
        }
        
        if config:
            default_config.update(config)
        
        lama_config = LamaConfig(
            ldm_steps=default_config["ldm_steps"],
            ldm_sampler=default_config["ldm_sampler"],
            hd_strategy=default_config["hd_strategy"],
            hd_strategy_crop_margin=default_config["hd_strategy_crop_margin"],
            hd_strategy_crop_trigger_size=default_config["hd_strategy_crop_trigger_size"],
            hd_strategy_resize_limit=default_config["hd_strategy_resize_limit"]
        )
        
        # Convert BGR to RGB for LaMa
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run inpainting
        result = self.models["lama"](
            image=image_rgb,
            mask=mask,
            config=lama_config
        )
        
        # Convert back to BGR
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        return result_bgr
    
    def inpaint_with_stable_diffusion(self,
                                    image: np.ndarray,
                                    mask: np.ndarray,
                                    prompt: str = "",
                                    strength: float = 0.8,
                                    guidance_scale: float = 7.5) -> np.ndarray:
        """
        Inpainting using Stable Diffusion
        """
        if "sd" not in self.models:
            raise ValueError("Stable Diffusion model not loaded")
        
        # Convert to PIL
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        mask_pil = Image.fromarray(mask)
        
        # Run inpainting
        result = self.models["sd"](
            prompt=prompt,
            image=image_pil,
            mask_image=mask_pil,
            strength=strength,
            guidance_scale=guidance_scale
        ).images[0]
        
        # Convert back to numpy BGR
        result_np = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        return result_np
    
    def remove_text_and_translate(self,
                                image: np.ndarray,
                                text_bboxes: List[List[float]],
                                translated_texts: List[str],
                                character_bboxes: List[List[float]] = None,
                                method: str = "auto",
                                font_scale: float = 1.0,
                                font_color: Tuple[int, int, int] = (0, 0, 0),
                                font_thickness: int = 2) -> np.ndarray:
        """
        Complete pipeline: remove original text and add translated text
        
        Args:
            image: Input manga image
            text_bboxes: Bounding boxes of text regions
            translated_texts: List of translated texts
            character_bboxes: Character bounding boxes to preserve
            method: Inpainting method ("opencv", "lama", "sd", "auto")
            font_scale: Scale for text rendering
            font_color: Color for translated text (BGR)
            font_thickness: Thickness for text rendering
        
        Returns:
            Image with original text removed and translations added
        """
        # Step 1: Create mask for text removal
        mask = self.create_text_removal_mask(image, text_bboxes, character_bboxes)
        
        # Step 2: Inpaint the text regions
        if method == "auto":
            if "lama" in self.models:
                result_image = self.inpaint_with_lama(image, mask)
            else:
                result_image = self.inpaint_with_opencv(image, mask)
        elif method == "lama" and "lama" in self.models:
            result_image = self.inpaint_with_lama(image, mask)
        elif method == "sd" and "sd" in self.models:
            result_image = self.inpaint_with_stable_diffusion(image, mask)
        else:
            result_image = self.inpaint_with_opencv(image, mask)
        
        # Step 3: Add translated text
        result_with_text = self.add_translated_text(
            result_image, text_bboxes, translated_texts,
            font_scale, font_color, font_thickness
        )
        
        return result_with_text
    
    def add_translated_text(self,
                          image: np.ndarray,
                          text_bboxes: List[List[float]],
                          translated_texts: List[str],
                          font_scale: float = 1.0,
                          font_color: Tuple[int, int, int] = (0, 0, 0),
                          font_thickness: int = 2) -> np.ndarray:
        """
        Add translated text to the inpainted regions
        """
        result = image.copy()
        height, width = image.shape[:2]
        
        # Try to load a manga-appropriate font
        try:
            # You might need to adjust this path to your font file
            font = cv2.FONT_HERSHEY_SIMPLEX
        except:
            font = cv2.FONT_HERSHEY_SIMPLEX
        
        for bbox, text in zip(text_bboxes, translated_texts):
            if not text.strip():
                continue
                
            # Convert normalized coordinates to pixels if needed
            if all(0 <= v <= 1 for v in bbox):
                x1 = int(bbox[0] * width)
                y1 = int(bbox[1] * height)
                x2 = int(bbox[2] * width)
                y2 = int(bbox[3] * height)
            else:
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Calculate text position (center of bbox)
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            
            # Simple text wrapping
            words = text.split()
            lines = []
            current_line = []
            
            for word in words:
                test_line = ' '.join(current_line + [word])
                text_size = cv2.getTextSize(test_line, font, font_scale, font_thickness)[0]
                
                if text_size[0] <= bbox_width * 0.9:
                    current_line.append(word)
                else:
                    if current_line:
                        lines.append(' '.join(current_line))
                    current_line = [word]
            
            if current_line:
                lines.append(' '.join(current_line))
            
            # Calculate total text height
            line_height = cv2.getTextSize("Test", font, font_scale, font_thickness)[0][1]
            total_height = len(lines) * line_height
            
            # Draw each line
            start_y = y1 + (bbox_height - total_height) // 2 + line_height
            
            for i, line in enumerate(lines):
                text_size = cv2.getTextSize(line, font, font_scale, font_thickness)[0]
                text_x = x1 + (bbox_width - text_size[0]) // 2
                text_y = start_y + i * line_height
                
                # Add text with outline for better readability
                cv2.putText(result, line, (text_x, text_y), font, font_scale, 
                           (255, 255, 255), font_thickness + 2, cv2.LINE_AA)
                cv2.putText(result, line, (text_x, text_y), font, font_scale, 
                           font_color, font_thickness, cv2.LINE_AA)
        
        return result
    
    def image_to_base64(self, image: np.ndarray) -> str:
        """Convert numpy image to base64 string"""
        success, buffer = cv2.imencode('.png', image)
        if not success:
            raise ValueError("Could not encode image to PNG")
        
        img_str = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/png;base64,{img_str}"
    
    def base64_to_image(self, base64_str: str) -> np.ndarray:
        """Convert base64 string to numpy image"""
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        
        img_data = base64.b64decode(base64_str)
        img_array = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img

# Global instance
manga_inpainter = MangaInpainter()