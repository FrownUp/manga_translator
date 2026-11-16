import requests
import base64
import json
from PIL import Image
import io
import numpy as np

class MangaBackend:
    def __init__(self, detection_url="http://localhost:5000/detect", ocr_url="http://localhost:5000/ocr"):
        self.detection_url = detection_url
        self.ocr_url = ocr_url
    
    def detect_components(self, image_path):
        """Send image to YOLOv8 detection endpoint"""
        with open(image_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        
        response = requests.post(
            self.detection_url,
            json={"image": img_base64}
        )
        return response.json()
    
    def get_character_embeddings(self, image_path, character_bboxes):
        """Get MLP embeddings for characters"""
        with open(image_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        
        response = requests.post(
            "http://localhost:5000/embeddings",
            json={
                "image": img_base64,
                "bboxes": character_bboxes
            }
        )
        return response.json()["embeddings"]
    
    def ocr_text_regions(self, image_path, text_bboxes):
        """Get OCR results for text regions"""
        with open(image_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        
        response = requests.post(
            self.ocr_url,
            json={
                "image": img_base64,
                "bboxes": text_bboxes
            }
        )
        return response.json()["results"]
    
    def base64_to_image(self, base64_str):
        """Convert base64 to PIL Image"""
        img_data = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(img_data))
    
    def image_to_base64(self, image):
        """Convert PIL Image to base64"""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')