# magi_server.py - IMPROVED ERROR HANDLING
from detection_processor import convert_yolo_to_magi_format, sort_panels
from ocr_processor import OCRFusion, crop_image, clean_ocr_text
from relation_predictor import predict_relations
from relation_predictor import extract_geometric_features, convert_box
from association_engine import cluster_characters, associate_text_to_characters
from transcript_generator import generate_transcript
from manga_pipeline import RelationClassifier
from character_mapper import CharacterMapper
import os
from translator import (
    translate_text,
    batch_translate,
    get_translation_stats,
    clear_translation_memory,
    ContextAwareMangaProcessor,
    ReadingOrder,
    get_translation_stats
) 
import asyncio
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ✅ FIXED: Use only expandable_segments, remove conflicting setting
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── model paths (auto-patched by setup_local.py) ──
from model_paths import (
    MODEL_DIR,
    YOLO_BUBBLE_MODEL_PATH,
    YOLO_PANEL_MODEL_PATH,
    YOLO_FACE_MODEL_PATH,
    EMBED_MODEL_PATH,
    CLASSIFIER_MODEL_PATH,
)



from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import torch
import base64
from io import BytesIO
from PIL import Image
from ultralytics import YOLO
from torch import nn
import joblib
from transformers import DonutProcessor, VisionEncoderDecoderModel
import requests
import re
import cairosvg
import json
import io
from datetime import datetime
import gc
from website_structure_analyzer import WebsiteStructureAnalyzer
from virtual_manga_page import ContextAwareMangaProcessor, ReadingOrder
from integrated_translator import setup_enhanced_translator_endpoints, EnhancedMangaTranslator

# ✅ FIXED: Remove unused flask_cors import
# from flask_cors import CORS  # Not needed - using FastAPI CORSMiddleware
from image_capture import image_capture
from realtime_overlay import realtime_overlay
from bulk_processor import bulk_processor
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

try:
    from bubble_tracker import bubble_tracker
    BUBBLE_TRACKING_AVAILABLE = True
    print("Bubble tracking enabled")
except ImportError as e:
    BUBBLE_TRACKING_AVAILABLE = False
    bubble_tracker = None
    print(f"Bubble tracking disabled: {e}")

# Initialize inpainter
try:
    from manga_inpainter import MangaInpainter
    manga_inpainter = MangaInpainter()
    INPAINTING_AVAILABLE = True
    print("Manga inpainting system initialized")
except ImportError as e:
    INPAINTING_AVAILABLE = False
    manga_inpainter = None
    print(f"Inpainting system disabled: {e}")

torch.cuda.empty_cache()
gc.collect()

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from functools import lru_cache
import uvicorn
from concurrent.futures import ThreadPoolExecutor
import time
from starlette.concurrency import run_in_threadpool

app = FastAPI(title="MAGI Server Optimized")

# ---------------------------------------------------------------------------
# SPEED: keep inference OFF the event loop, serialize only the GPU stage.
#
# Before: every handler was `async def` but ran YOLO / OCR / base64 decoding directly on the
# event loop, so the whole server handled ONE request at a time (and could not even answer
# /health while busy). Now: decoding runs in worker threads in parallel, GPU/model work goes
# through one FIFO async lock (models are not thread-safe), and a request whose client has already
# disconnected (e.g. the browser aborted it) is skipped before it ever reaches the GPU.
# ---------------------------------------------------------------------------
_GPU_ALOCK = None

def _gpu_alock():
    global _GPU_ALOCK
    if _GPU_ALOCK is None:          # created lazily so it binds to uvicorn's running loop
        _GPU_ALOCK = asyncio.Lock()
    return _GPU_ALOCK

class _ClientGone(Exception):
    pass

async def _gpu_stage(request, fn, *args, **kwargs):
    async with _gpu_alock():
        try:
            if await request.is_disconnected():
                raise _ClientGone()
        except _ClientGone:
            raise
        except Exception:
            pass                    # cannot tell -> just run it
        return await run_in_threadpool(fn, *args, **kwargs)

def _extract_feats(bboxes, img_width, img_height):
    feats = []
    for bbox in bboxes:
        if isinstance(bbox, dict):          # accept detection objects as well as plain [x1,y1,x2,y2]
            bbox = bbox.get("bbox")
        x1 = int(bbox[0] * img_width)
        y1 = int(bbox[1] * img_height)
        x2 = int(bbox[2] * img_width)
        y2 = int(bbox[3] * img_height)
        center_x = (x1 + x2) / 2 / img_width
        center_y = (y1 + y2) / 2 / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        feats.append([center_x, center_y, width, height])
    return feats

def _compute_embeddings(text_bboxes, char_bboxes, img_width, img_height):
    text_feats = _extract_feats(text_bboxes, img_width, img_height)
    char_feats = _extract_feats(char_bboxes, img_width, img_height)
    if not text_feats or not char_feats:
        return {"text_embeddings": [], "char_embeddings": []}
    text_tensor = torch.tensor(text_feats, dtype=torch.float32)
    char_tensor = torch.tensor(char_feats, dtype=torch.float32)
    with torch.no_grad():
        text_embeddings, char_embeddings = EMBED_MODEL(text_tensor, char_tensor)
        return {
            "text_embeddings": text_embeddings.cpu().numpy().tolist(),
            "char_embeddings": char_embeddings.cpu().numpy().tolist()
        }

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Remove Flask imports - not needed
# flask_app = Flask(__name__)
# CORS(flask_app)

# ✅ IN-MEMORY CACHING LAYER (128 item LRU)
@lru_cache(maxsize=128)
def cached_process_image(image_hash: str):
    pass

website_analyzer = WebsiteStructureAnalyzer()

manga_processor = ContextAwareMangaProcessor()

enhanced_translator = EnhancedMangaTranslator()

# Add this after your existing Flask routes
setup_enhanced_translator_endpoints(app)

# -----------------------------------------
# Model Paths
# -----------------------------------------
YOLO_BUBBLE_MODEL_PATH = os.path.join(MODEL_DIR, "bubble_best.pt")
YOLO_PANEL_MODEL_PATH = os.path.join(MODEL_DIR, "panel_best.pt")
YOLO_FACE_MODEL_PATH = os.path.join(MODEL_DIR, "face_best.pt")
# -----------------------------------------
# LAZY MODEL LOADING - Load only when needed
# -----------------------------------------
class LazyModelLoader:
    """Load models on-demand to speed up startup"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._models = {}
            cls._instance._loading = {}
        return cls._instance
    
    def get_model(self, model_name):
        """Get model, loading it if necessary"""
        if model_name not in self._models:
            if model_name in self._loading:
                # Wait for model being loaded by another thread
                import time
                while model_name in self._loading:
                    time.sleep(0.1)
                return self._models.get(model_name)
            
            # Mark as loading
            self._loading[model_name] = True
            
            try:
                print(f"Loading model: {model_name}")
                if model_name == 'face_detection':
                    model = YOLO(YOLO_FACE_MODEL_PATH, task="detect")
                    model.conf = 0.25
                    model.iou = 0.45
                    model.verbose = False
                elif model_name == 'bubble_char':
                    model = YOLO(YOLO_BUBBLE_MODEL_PATH, task="detect")
                    model.conf = 0.3
                    model.iou = 0.4
                    model.verbose = False
                elif model_name == 'panel':
                    model = YOLO(YOLO_PANEL_MODEL_PATH, task="detect")
                    model.conf = 0.35
                    model.iou = 0.4
                    model.verbose = False
                else:
                    raise ValueError(f"Unknown model: {model_name}")
                
                self._models[model_name] = model
                print(f"Model loaded: {model_name}")
            finally:
                self._loading.pop(model_name, None)
        
        return self._models[model_name]

# Global model loader instance
model_loader = LazyModelLoader()

# Helper functions to get models
def get_face_detection_model():
    return model_loader.get_model('face_detection')

def get_bubble_char_model():
    return model_loader.get_model('bubble_char')

def get_panel_model():
    return model_loader.get_model('panel')

class EmbeddingSGGModel(nn.Module):
    def __init__(self, input_dim=4, embed_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )
    def forward(self, text_feats, char_feats):
        text_embed = self.encoder(text_feats)
        char_embed = self.encoder(char_feats)
        return text_embed, char_embed

EMBED_MODEL_PATH = os.path.join(MODEL_DIR, "sgg_relation_model.pth")
EMBED_MODEL = EmbeddingSGGModel(input_dim=4, embed_dim=32)
EMBED_MODEL.load_state_dict(torch.load(EMBED_MODEL_PATH))
EMBED_MODEL.eval()

# Use single instance of relation classifier
classifier_model = RelationClassifier(input_dim=10)
classifier_model.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH))
classifier_model.eval()

ocr_fusion_model = OCRFusion()

DATASET_PATH = "ocr_data_set"
os.makedirs(DATASET_PATH, exist_ok=True)

# -----------------------------------------
# Utils
# -----------------------------------------
def detect_faces(img):
    model = get_face_detection_model()  # Lazy load
    results = model(img)[0]
    face_bboxes = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        face_bboxes.append([x1, y1, x2, y2])
    return face_bboxes

def run_async(func, *args, **kwargs):
    """Safely run async functions in sync context (kept for compatibility)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Run inside existing loop
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(func(*args, **kwargs))
    else:
        # Normal case
        return asyncio.run(func(*args, **kwargs))

# ✅ FIXED: Remove duplicate asyncio import
# import asyncio  # Already imported at top

def base64_to_cv2(image_base64: str) -> np.ndarray:
    if not image_base64 or not isinstance(image_base64, str):
        raise ValueError("Empty or invalid base64 string")
    
    # Remove data URL prefix if present
    if ',' in image_base64:
        _, image_base64 = image_base64.split(',', 1)
    
    image_base64 = image_base64.strip()
    image_base64 = re.sub(r'\s+', '', image_base64)
    
    # Add padding if needed
    missing_padding = len(image_base64) % 4
    if missing_padding:
        image_base64 += '=' * (4 - missing_padding)
    
    try:
        img_data = base64.b64decode(image_base64)
    except Exception as e:
        raise ValueError(f"Base64 decoding failed: {e}")

    # Detect SVG
    if img_data.lstrip().startswith(b'<svg'):
        try:
            png_bytes = cairosvg.svg2png(bytestring=img_data)
            img_pil = Image.open(BytesIO(png_bytes))
            img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            return img
        except Exception as e:
            raise ValueError(f"SVG to PNG conversion failed: {e}")
    
    try:
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)     # fast path (handles RGBA PNG from canvas)
        if img is not None:
            return img
    except Exception:
        pass

    try:
        img_pil = Image.open(BytesIO(img_data))
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return img
    except Exception:
        pass
    
    try:
        with open("debug_image.jpg", "wb") as f:
            f.write(img_data)
        img = cv2.imread("debug_image.jpg")
        if img is not None:
            return img
    except Exception:
        pass
    
    raise ValueError("All image decoding methods failed")

_FREE_EVERY = 25
_free_calls = 0

def free_memory(*objs):
    """Drop references now; run the expensive CUDA-cache flush + full GC only every _FREE_EVERY calls.
    (empty_cache() forces the allocator to re-grab VRAM on the next request, and gc.collect() is a
    full collection - doing both after every single request cost far more than it saved.)"""
    global _free_calls
    for obj in objs:
        if obj is not None:
            try:
                del obj
            except:
                pass
    _free_calls += 1
    if _free_calls % _FREE_EVERY == 0:
        torch.cuda.empty_cache()
        gc.collect()

def validate_image_data(image_data):
    """Validate image data before processing"""
    if not image_data or not isinstance(image_data, str):
        return False, "No image data provided"
    
    if len(image_data) < 100:  # Too short to be a valid image
        return False, "Image data too short"
    
    # Check if it looks like base64
    if not re.match(r'^[A-Za-z0-9+/]*={0,2}$', image_data.split(',')[-1]):
        return False, "Invalid base64 format"
    
    return True, "Valid"

# ✅ GPU BATCH PROCESSING UTILITY
@torch.inference_mode()
def run_detection_batch(images, batch_size=4):
    """Run detection on multiple images in batch on GPU"""
    if not images:
        return []
    
    all_results = []
    
    # Process in batches
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        
        # Run all 3 models in parallel on same batch
        panel_batch_results = PANEL_MODEL(batch, verbose=False, half=True)
        bubble_batch_results = BUBBLE_CHAR_MODEL(batch, verbose=False, half=True)
        
        for idx, img in enumerate(batch):
            height, width = img.shape[:2]
            
            def bbox_xyxy_normalized(box):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                return [x1/width, y1/height, x2/width, y2/height]
            
            # Panels
            panels = []
            for box in panel_batch_results[idx].boxes:
                if int(box.cls) == 0:
                    panels.append({"bbox": bbox_xyxy_normalized(box), "confidence": float(box.conf)})
            
            # Bubbles and characters
            bubbles = []
            characters = []
            for box in bubble_batch_results[idx].boxes:
                cls_id = int(box.cls)
                item = {"bbox": bbox_xyxy_normalized(box), "confidence": float(box.conf)}
                if cls_id == 0:
                    bubbles.append(item)
                elif cls_id == 1:
                    characters.append(item)
            
            all_results.append({
                "panels": panels or [],
                "bubbles": bubbles or [],
                "characters": characters or []
            })
    
    return all_results


def run_detection(img):
    height, width = img.shape[:2]

    def bbox_xyxy_normalized(box):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        return [x1/width, y1/height, x2/width, y2/height]

    # ✅ LAZY LOAD MODELS + FP16
    with torch.amp.autocast('cuda', dtype=torch.float16, enabled=True):
        panel_model = get_panel_model()
        bubble_char_model = get_bubble_char_model()
        
        panel_results = panel_model(img, verbose=False)[0]
        panels = [{"bbox": bbox_xyxy_normalized(box), "confidence": float(box.conf)}
                  for box in panel_results.boxes if int(box.cls) == 0]

        # Bubbles and characters
        bubble_char_results = bubble_char_model(img, verbose=False)[0]
        bubbles = []
        characters = []
        for box in bubble_char_results.boxes:
            cls_id = int(box.cls)
            item = {"bbox": bbox_xyxy_normalized(box), "confidence": float(box.conf)}
            if cls_id == 0:
                bubbles.append(item)
            elif cls_id == 1:
                characters.append(item)

    return {
        "panels": panels or [],
        "bubbles": bubbles or [],
        "characters": characters or []
    }

def format_yolo_boxes(results, cls_type, image_shape):
    """Extract bounding boxes for a specific class in pixel format."""
    boxes = []
    for box in results.boxes:
        if hasattr(box, 'cls') and int(box.cls) == cls_type:
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # left, top, right, bottom
            boxes.append({"bbox": [x1, y1, x2, y2]})
    return boxes

def fetch_image_base64_via_proxy(image_url: str, proxy_url="http://localhost:3000/api/image-to-base64") -> str:
    """
    Given an image URL, fetch raw base64 image string via your Node.js CORS proxy.
    """
    response = requests.post(proxy_url, json={"imageUrl": image_url})
    response.raise_for_status()
    data = response.json()
    if "base64" not in data:
        raise ValueError("No base64 key in proxy response")
    return data["base64"]  # raw base64 string without header prefix

@app.post('/capture_image')
async def capture_image(request: Request):
    """Capture image from webpage using Selenium"""
    try:
        data = await request.json()
        if not data:
            return JSONResponse(status_code=400, content={"error": "No JSON data provided"})
            
        url = data.get('url')
        element_selector = data.get('element_selector')
        method = data.get('method', 'screenshot')
        
        if not url:
            return JSONResponse(status_code=400, content={"error": "No URL provided"})

        # Use webdriver-manager for automatic driver management
        from webdriver_manager.chrome import ChromeDriverManager
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.chrome.options import Options
        
        # Setup Chrome options
        chrome_options = Options()
        chrome_options.add_argument('--headless=new')  # Modern headless mode
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--disable-logging')
        chrome_options.add_argument('--log-level=3')
        
        # Initialize driver with webdriver-manager
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        try:
            driver.get(url)
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            target_element = None
            if element_selector:
                target_element = driver.find_element(By.CSS_SELECTOR, element_selector)
            
            # Perform capture based on method
            if method == 'canvas' and element_selector:
                result = image_capture.capture_from_canvas(driver, element_selector)
            else:
                result = image_capture.smart_capture(driver, target_element)
            
            if result['image'] is not None:
                # Convert image to base64
                result_base64 = image_capture.image_to_base64(result['image'])
                
                return JSONResponse(content={
                    "status": "success",
                    "captured_image": result_base64,
                    "method_used": result['method'],
                    "element_info": result['element_info']
                })
            else:
                return JSONResponse(status_code=400, content={"error": "Image capture failed"})
                
        finally:
            driver.quit()
            
    except Exception as e:
        print(f"Capture error: {str(e)}")
        return JSONResponse(status_code=400, content={"error": f"Capture failed: {str(e)}"})


@app.post('/create_overlay')
async def create_overlay(request: Request):
    """Create text overlay using backend's own OCR results"""
    try:
        data = await request.json()
        image_data = data.get('image')
        text_elements = data.get('text_elements', [])
        
        print(f"Received {len(text_elements)} text elements for overlay")
        
        img = base64_to_cv2(image_data)
        
        # If we have bounding boxes but no reliable text, RE-OCR the image
        needs_reocr = any(
            not elem.get('original_text') or 
            not elem.get('original_text').strip() or
            '□' in elem.get('original_text', '')  # Square boxes detected
            for elem in text_elements
        )
        
        if needs_reocr:
            print("Some texts are missing or corrupted, re-OCRing...")
            # Extract just the bounding boxes for re-OCR
            bboxes = [elem['bbox'] for elem in text_elements]
            
            # Re-OCR the specific regions
            crops = [crop_image(img, bbox) for bbox in bboxes]
            ocr_results = ocr_fusion_model.best_ocr(crops, lang='jpn')  # or detect lang
            
            # Replace corrupted/missing texts with fresh OCR results
            for i, (elem, ocr_text) in enumerate(zip(text_elements, ocr_results)):
                if not elem.get('original_text') or not elem.get('original_text').strip() or '□' in elem.get('original_text', ''):
                    print(f"Replacing corrupted text '{elem.get('original_text')}' with '{ocr_text}'")
                    elem['original_text'] = ocr_text
        
        # Use the (now corrected) original texts for overlay
        text_bboxes = [elem['bbox'] for elem in text_elements]
        display_texts = []
        styles = []
        
        for elem in text_elements:
            # Use original text from OCR (now guaranteed to be valid)
            original_text = elem.get('original_text', '')
            translated_text = elem.get('translated_text', '')
            
            # Debug: check text content
            print(f"Overlay text - Original: '{original_text[:30]}...', Translated: '{translated_text[:30]}...'")
            
            # Choose text to display
            if translated_text and translated_text.strip() and translated_text != original_text:
                display_text = translated_text
            else:
                display_text = original_text
                
            display_texts.append(display_text)
            styles.append(elem.get('style', realtime_overlay._get_default_style()))
        
        # Create overlay with guaranteed valid text
        result_image = realtime_overlay.create_text_overlay(
            img, text_bboxes, display_texts, styles
        )
        
        result_base64 = manga_inpainter.image_to_base64(result_image)
        
        return JSONResponse(content={
            'overlay_image': result_base64,
            'success': True,
            'debug': {
                'texts_used': [t[:50] + '...' for t in display_texts],
                'reocr_performed': needs_reocr
            }
        })
        
    except Exception as e:
        print(f"Overlay creation error: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={'error': str(e), 'success': False})

@app.post('/process_batch')
async def process_batch(request: Request):
    """Process multiple images in batch"""
    try:
        data = await request.json()
        if not data:
            return JSONResponse(status_code=400, content={'error': 'No valid JSON data provided', 'status': 'failed'})
            
        image_paths = data.get('image_paths', [])
        output_dir = data.get('output_dir', 'batch_output')
        options = data.get('options', {})
        
        # Debug logging
        logger.info(f"[process_batch] Received request: {len(image_paths)} files, output_dir={output_dir}")
        
        # Validate files
        validation = bulk_processor.validate_input_files(image_paths)
        
        if validation['total_valid'] == 0:
            return JSONResponse(status_code=400, content={
                'error': 'No valid files to process',
                'validation': validation,
                'status': 'failed'
            })
        
        # Process batch - bulk_processor.process_batch is already async
        result = await bulk_processor.process_batch(
            validation['valid_files'], 
            output_dir, 
            options
        )
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"[process_batch] ERROR: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={'error': f'Batch processing failed: {str(e)}', 'status': 'failed'})

@app.api_route('/system_status', methods=['GET', 'POST'])
async def system_status(request: Request):
    """Get comprehensive system status"""
    status = {
        'inpainting': INPAINTING_AVAILABLE,
        'bulk_processing': True,
        'realtime_overlay': True,  # Set to True since we have the endpoint
        'translation_analytics': {},  # Use empty dict instead of context_aware_translator
        'supported_formats': ['png', 'jpg', 'jpeg', 'webp'],
        'backend_healthy': True,
        'models_loaded': True,
        'available_services': ['detect', 'ocr', 'embeddings', 'translate', 'create_overlay']
    }
    
    return JSONResponse(content=status)

@app.post('/inpaint_text')
async def inpaint_text(request: Request):
    """Remove text from manga and optionally add translations"""
    try:
        data = await request.json()
        if not data:
            return JSONResponse(status_code=400, content={"error": "No JSON data provided"})
            
        image_data = data.get('image')
        if not image_data:
            return JSONResponse(status_code=400, content={"error": "No image data provided"})
            
        # Validate and decode image
        is_valid, message = validate_image_data(image_data)
        if not is_valid:
            return JSONResponse(status_code=400, content={"error": f"Invalid image data: {message}"})
            
        img = base64_to_cv2(image_data)
        
        # Get text bounding boxes
        text_bboxes = data.get('text_bboxes', [])
        if not text_bboxes:
            return JSONResponse(status_code=400, content={"error": "No text bounding boxes provided"})
            
        # Optional parameters
        character_bboxes = data.get('character_bboxes', [])
        translated_texts = data.get('translated_texts', [])
        method = data.get('method', 'auto')
        add_translations = data.get('add_translations', False)
        
        if add_translations and len(translated_texts) != len(text_bboxes):
            return JSONResponse(status_code=400, content={"error": "Number of translated texts must match text bounding boxes"})
        
        # Perform inpainting
        if add_translations:
            result_image = manga_inpainter.remove_text_and_translate(
                img, text_bboxes, translated_texts, character_bboxes, method
            )
        else:
            mask = manga_inpainter.create_text_removal_mask(img, text_bboxes, character_bboxes)
            
            if method == "lama" and "lama" in manga_inpainter.models:
                result_image = manga_inpainter.inpaint_with_lama(img, mask)
            elif method == "sd" and "sd" in manga_inpainter.models:
                result_image = manga_inpainter.inpaint_with_stable_diffusion(img, mask)
            else:
                result_image = manga_inpainter.inpaint_with_opencv(img, mask)
        
        # Convert result to base64
        result_base64 = manga_inpainter.image_to_base64(result_image)
        
        return JSONResponse(content={
            "status": "success",
            "result_image": result_base64,
            "method_used": method,
            "text_regions_processed": len(text_bboxes)
        })
        
    except Exception as e:
        app.logger.error(f"Inpainting error: {str(e)}")
        return JSONResponse(status_code=400, content={"error": f"Inpainting failed: {str(e)}"})

@app.post('/create_mask')
async def create_mask(request: Request):
    """Create inpainting mask from bounding boxes"""
    try:
        data = await request.json()
        image_data = data.get('image')
        text_bboxes = data.get('text_bboxes', [])
        character_bboxes = data.get('character_bboxes', [])
        exclude_characters = data.get('exclude_characters', True)
        
        img = base64_to_cv2(image_data)
        
        mask = manga_inpainter.create_text_removal_mask(
            img, text_bboxes, character_bboxes, exclude_characters
        )
        
        mask_base64 = manga_inpainter.image_to_base64(mask)
        
        return JSONResponse(content={
            "status": "success",
            "mask_image": mask_base64,
            "mask_area_pixels": int(np.sum(mask == 255))
        })
        
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.get('/inpainting_status')
async def inpainting_status():
    """Get status of available inpainting models"""
    models_status = {
        "opencv": "opencv" in manga_inpainter.models,
        "lama": "lama" in manga_inpainter.models,
        "stable_diffusion": "sd" in manga_inpainter.models,
        "device": manga_inpainter.device
    }
    
    return JSONResponse(content=models_status)

@app.post('/process_with_inpainting')
async def process_with_inpainting(request: Request):
    """Complete pipeline: detect, translate, and inpaint with translations"""
    try:
        data = await request.json()
        image_data = data.get('image')
        
        # Step 1: Run detection
        img = base64_to_cv2(image_data)
        detections = run_detection(img)
        
        # Step 2: Run OCR on text regions
        text_bboxes = [item["bbox"] for item in detections.get("texts", [])]
        text_crops = [crop_image(img, bbox) for bbox in text_bboxes]
        ocr_results = ocr_fusion_model.best_ocr(text_crops)
        
        # Step 3: Translate texts
        translated_texts = []
        for text in ocr_results:
            translated, _ = await translate_text(text)
            translated_texts.append(translated)
        
        # Step 4: Inpaint and add translations
        character_bboxes = [item["bbox"] for item in detections.get("characters", [])]
        
        result_image = manga_inpainter.remove_text_and_translate(
            img, text_bboxes, translated_texts, character_bboxes
        )
        
        result_base64 = manga_inpainter.image_to_base64(result_image)
        
        return JSONResponse(content={
            "status": "success",
            "result_image": result_base64,
            "detections": detections,
            "ocr_results": ocr_results,
            "translated_texts": translated_texts
        })
        
    except Exception as e:
        app.logger.error(f"Full pipeline with inpainting error: {str(e)}")
        return JSONResponse(status_code=400, content={"error": f"Processing failed: {str(e)}"})

@app.post('/process_with_enhanced_analysis')
async def process_with_enhanced_analysis(request: Request):
    data = await request.json()
    image_path = data['image_path']
    detection_results = data['detection_results']
    
    # Create virtual page
    virtual_page = manga_processor.create_virtual_page_from_detection(
        image_path, detection_results, ReadingOrder.MANGA
    )
    
    # Process each bubble with enhanced analysis
    enhanced_results = []
    for i, bubble in enumerate(virtual_page.bubbles):
        bubble_id = f"bubble_{i}"
        
        # Use enhanced translation with analysis
        result = await enhanced_translator.translate_with_full_analysis(
            bubble.original_text,
            bubble.language,
            bubble_id
        )
        
        enhanced_results.append(result)
        
        # Update virtual page with analyzed translation
        manga_processor.update_bubble_translation(
            bubble.id, 
            result['translation_result']['translated_text'],
            result['overall_assessment']['quality_score']
        )
    
    return JSONResponse(content={
        "virtual_page": virtual_page.export_to_json(),
        "translation_analytics": enhanced_translator.get_system_analytics(),
        "enhanced_results": enhanced_results
    })



@app.post('/translate_with_context')
async def translate_with_context(request: Request):
    """Enhanced translation endpoint with context support"""
    try:
        data = await request.json()
        if not data:
            return JSONResponse(status_code=400, content={"error": "No JSON data provided"})
            
        text = data.get('text', '')
        source_lang = data.get('source_language', 'auto')
        bubble_id = data.get('bubble_id')
        context_bubbles = data.get('context_bubbles', [])
        speaker = data.get('speaker')
        
        if not text:
            return JSONResponse(status_code=400, content={"error": "No text provided"})
        
        # Run async translation with context
        translated_text, detected_lang = await translate_text(
            text, source_lang, bubble_id
        )
        
        return JSONResponse(content={
            "original_text": text,
            "translated_text": translated_text,
            "detected_language": detected_lang,
            "source_language": source_lang,
            "bubble_id": bubble_id,
            "context_used": len(context_bubbles) > 0
        })
        
    except Exception as e:
        app.logger.error(f"Context-aware translation error: {str(e)}")
        return JSONResponse(status_code=400, content={"error": f"Translation failed: {str(e)}"})

@app.post('/setup_virtual_page')
async def setup_virtual_page(request: Request):
    """Set up virtual page from detection results"""
    try:
        data = await request.json()
        image_path = data.get('image_path')
        detection_results = data.get('detection_results')
        reading_order = data.get('reading_order', 'manga')
        
        if not image_path or not detection_results:
            return JSONResponse(status_code=400, content={"error": "image_path and detection_results required"})
        
        virtual_page_json = setup_virtual_page_from_detection(
            image_path, detection_results, reading_order
        )
        
        return JSONResponse(content={
            "status": "success" if virtual_page_json else "failed",
            "virtual_page": virtual_page_json
        })
        
    except Exception as e:
        app.logger.error(f"Virtual page setup error: {str(e)}")
        return JSONResponse(status_code=400, content={"error": f"Virtual page setup failed: {str(e)}"})

@app.get('/get_translation_context/{bubble_id}')
async def get_translation_context_route(bubble_id: str):
    """Get context for a specific bubble"""
    try:
        context = manga_processor.get_context_for_translation(bubble_id)
        return JSONResponse(content={
            "bubble_id": bubble_id,
            "context_bubbles": context
        })
    except Exception as e:
        app.logger.error(f"Get context error: {str(e)}")
        return JSONResponse(status_code=400, content={"error": f"Failed to get context: {str(e)}"})


@app.post('/process_with_context')
async def process_with_context(request: Request):
    """Create virtual page and process each bubble with context-aware translation"""
    try:
        data = await request.json()
        if not data:
            return JSONResponse(status_code=400, content={"error": "No JSON data provided"})

        image_path = data['image_path']
        detection_results = data['detection_results']

        # Create virtual page
        virtual_page = manga_processor.create_virtual_page_from_detection(
            image_path, detection_results, ReadingOrder.MANGA
        )

        # Process and translate each bubble
        for bubble in virtual_page.bubbles:
            # Async translation
            translated_text, _ = await translate_text(
                bubble.original_text,
                bubble.language,
                bubble.id
            )

            # Update virtual page with translation
            manga_processor.update_bubble_translation(bubble.id, translated_text)

        return JSONResponse(content={
            "virtual_page": virtual_page.export_to_json(),
            "reading_analysis": virtual_page.analyze_reading_patterns()
        })

    except Exception as e:
        app.logger.error(f"Process with context error: {str(e)}")
        return JSONResponse(status_code=400, content={"error": f"Processing failed: {str(e)}"})


@app.post('/translate')
async def translate_endpoint(request: Request):
    """Translate a single text"""
    try:
        data = await request.json()
        if not data:
            return JSONResponse(status_code=400, content={"error": "No JSON data provided"})

        text = data.get('text', '')
        source_lang = data.get('source_language', 'auto')

        if not text:
            return JSONResponse(status_code=400, content={"error": "No text provided"})

        translated_text, detected_lang = await translate_text(text, source_lang)

        return JSONResponse(content={
            "original_text": text,
            "translated_text": translated_text,
            "detected_language": detected_lang,
            "source_language": source_lang
        })

    except Exception as e:
        app.logger.error(f"Translation error: {str(e)}")
        return JSONResponse(status_code=400, content={"error": f"Translation failed: {str(e)}"})


@app.post('/translate_batch')
async def translate_batch_endpoint(request: Request):
    """Batch translate texts"""
    try:
        data = await request.json()
        if not data:
            return JSONResponse(status_code=400, content={"error": "No JSON data provided"})

        texts = data.get('texts', [])
        source_lang = data.get('source_language', 'auto')

        if not texts:
            return JSONResponse(status_code=400, content={"error": "No texts provided"})

        translated_texts = await batch_translate(texts, source_lang)

        return JSONResponse(content={
            "original_texts": texts,
            "translated_texts": translated_texts,
            "source_language": source_lang
        })

    except Exception as e:
        app.logger.error(f"Batch translation error: {str(e)}")
        return JSONResponse(status_code=400, content={"error": f"Batch translation failed: {str(e)}"})


@app.get('/translation_stats')
async def translation_stats():
    """Get translation statistics"""
    return JSONResponse(content=get_translation_stats())


@app.post('/clear_translation_cache')
async def clear_cache():
    """Clear translation cache"""
    clear_translation_memory()
    return JSONResponse(content={"status": "Translation cache cleared"})

@app.post('/register_bubble')
async def register_bubble(request: Request):
    data = await request.json()
    bubble_id = bubble_tracker.register_bubble(
        original_text=data['original_text'],
        bbox=data['bbox'],
        container_info=data['container_info'],
        image_src=data['image_src'],
        source_language=data['source_language'],
        speaker=data.get('speaker')
    )
    return JSONResponse(content={"bubble_id": bubble_id})

@app.get('/bubble_stats')
async def get_bubble_stats():
    return JSONResponse(content=bubble_tracker.get_bubble_stats())

@app.get('/character_dialogue/{character_id}')
async def get_character_dialogue(character_id: str):
    dialogue = bubble_tracker.get_character_dialogue(character_id)
    return JSONResponse(content={"character_id": character_id, "dialogue": dialogue})

@app.post('/analyze_website')
async def analyze_website(request: Request):
    data = await request.json()
    url = data.get('url')
    
    if not url:
        return JSONResponse(status_code=400, content={"error": "URL is required"})
    
    try:
        result = website_analyzer.generate_overlay_data(url)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post('/get_element_path')
async def get_element_path(request: Request):
    data = await request.json()
    x = data.get('x')
    y = data.get('y')
    
    if x is None or y is None:
        return JSONResponse(status_code=400, content={"error": "X and Y coordinates are required"})
    
    try:
        path = website_analyzer.get_element_path(x, y)
        return JSONResponse(content={"element_path": path})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# -----------------------------------------
# FastAPI Endpoints - FIXED
# -----------------------------------------
@app.post('/detect')
async def detect(request: Request):
    img = None
    try:
        data = await request.json()
        if not data:
            return JSONResponse(content={"error": "No JSON data provided"}, status_code=400)

        image_data = data.get('image')
        is_valid, message = await run_in_threadpool(validate_image_data, image_data)
        if not is_valid:
            return JSONResponse(content={"error": f"Invalid image data: {message}"}, status_code=400)

        img = await run_in_threadpool(base64_to_cv2, image_data)      # parallel CPU work

        # Validate image dimensions
        if img.shape[0] < 10 or img.shape[1] < 10:
            return JSONResponse(content={"error": "Image too small"}, status_code=400)

        detections = await _gpu_stage(request, run_detection, img)    # serialized GPU work
        return JSONResponse(content=detections)

    except _ClientGone:
        return JSONResponse(content={"error": "client disconnected"}, status_code=499)
    except Exception as e:
        print(f"Detection error: {str(e)}")
        return JSONResponse(content={"error": f"Processing failed: {str(e)}"}, status_code=400)

    finally:
        if img is not None:
            free_memory(img)

@app.post('/process_url_image')
async def process_url_image(request: Request):
    try:
        data = await request.json()
        image_url = data.get('image_url')
        if not image_url:
            return JSONResponse(status_code=400, content={"error": "Missing image_url"})

        # 1. Fetch raw base64 string from proxy
        raw_base64 = fetch_image_base64_via_proxy(image_url)

        # 2. Decode to OpenCV image using existing function
        img = base64_to_cv2(raw_base64)

        if img is None:
            return JSONResponse(status_code=400, content={"error": "Failed to decode image"})

        # 3. Run detection (reuse existing function)
        detections = run_detection(img)

        # 4. Return detections or any processing results
        return JSONResponse(content={
            "detections": detections,
            "message": "Image processed successfully"
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

DATASET_PATH = "ocr_data_set"
os.makedirs(DATASET_PATH, exist_ok=True)

@app.post('/correction')
async def handle_correction(request: Request):
    try:
        data = await request.json()
        
        # Create unique filename using timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
        
        # Extract data from request
        image_base64 = data['image']
        bbox = data['bbox']
        original_text = data['original_text']
        corrected_text = data['corrected_text']
        lang = data['lang']
        
        # Decode image
        img = base64_to_cv2(image_base64)
        
        # Crop text region using bbox [x1, y1, x2, y2]
        height, width = img.shape[:2]
        x1 = int(bbox[0] * width)
        y1 = int(bbox[1] * height)
        x2 = int(bbox[2] * width)
        y2 = int(bbox[3] * height)
        
        # Ensure valid dimensions
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        
        if x2 > x1 and y2 > y1:
            text_region = img[y1:y2, x1:x2]
            
            # Save image crop
            image_filename = f"{timestamp}.png"
            image_path = os.path.join(DATASET_PATH, image_filename)
            cv2.imwrite(image_path, text_region)
            
            # Save metadata
            metadata = {
                "image_file": image_filename,
                "original_text": original_text,
                "corrected_text": corrected_text,
                "lang": lang,
                "bbox": bbox,
                "timestamp": timestamp
            }
            
            meta_filename = f"{timestamp}.json"
            meta_path = os.path.join(DATASET_PATH, meta_filename)
            
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return JSONResponse(content={
                "status": "success",
                "message": f"Correction saved to {DATASET_PATH}",
                "files": [image_filename, meta_filename]
            })
        else:
            return JSONResponse(status_code=400, content={
                "status": "error",
                "message": f"Invalid bounding box: {x1},{y1} to {x2},{y2}"
            })
    
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "status": "error",
            "message": str(e)
        })

@app.post('/embeddings')
async def embeddings(request: Request):
    img = None
    try:
        data = await request.json()
        if not data:
            return JSONResponse(content={"error": "No JSON data provided"}, status_code=400)

        # The embedding model only looks at normalized box geometry, so the pixels are not needed:
        # send "image_size": [width, height] and skip the multi-MB image upload + decode entirely.
        size = data.get('image_size')
        if isinstance(size, (list, tuple)) and len(size) == 2 and size[0] and size[1] and size[0] > 0 and size[1] > 0:
            img_width, img_height = int(size[0]), int(size[1])
        else:
            image_data = data.get('image')
            is_valid, message = await run_in_threadpool(validate_image_data, image_data)
            if not is_valid:
                return JSONResponse(content={"error": f"Invalid image data: {message}"}, status_code=400)
            img = await run_in_threadpool(base64_to_cv2, image_data)
            img_height, img_width = img.shape[:2]

        text_bboxes = data.get('text_bboxes', [])
        char_bboxes = data.get('char_bboxes', [])

        result = await run_in_threadpool(_compute_embeddings, text_bboxes, char_bboxes, img_width, img_height)
        return JSONResponse(content=result)

    except Exception as e:
        print(f"Embeddings error: {str(e)}")
        return JSONResponse(content={"error": f"Processing failed: {str(e)}"}, status_code=400)

    finally:
        if img is not None:
            free_memory(img)

@app.post('/ocr_and_overlay')
async def ocr_and_overlay(request: Request):
    """Complete workflow: OCR text and create overlay in one call"""
    try:
        data = await request.json()
        image_data = data.get('image')
        bboxes = data.get('bboxes', [])  # Text regions detected by object detection
        lang = data.get('lang', 'jpn')
        
        img = base64_to_cv2(image_data)
        
        if not bboxes:
            return JSONResponse(status_code=400, content={"error": "No bounding boxes provided"})
        
        # Step 1: OCR the text regions
        print(f"OCRing {len(bboxes)} text regions...")
        crops = [crop_image(img, bbox) for bbox in bboxes]
        ocr_results = ocr_fusion_model.best_ocr(crops, lang=lang)
        
        # Step 2: Create overlay directly with OCR results
        display_texts = []
        styles = []
        
        for i, ocr_text in enumerate(ocr_results):
            print(f"OCR result {i}: '{ocr_text}'")
            display_texts.append(ocr_text)
            styles.append(realtime_overlay._get_default_style())
        
        # Create overlay with the original OCR text
        result_image = realtime_overlay.create_text_overlay(
            img, bboxes, display_texts, styles
        )
        
        result_base64 = manga_inpainter.image_to_base64(result_image)
        
        return JSONResponse(content={
            'overlay_image': result_base64,
            'ocr_results': ocr_results,
            'success': True
        })
        
    except Exception as e:
        print(f"OCR+Overlay error: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={'error': str(e), 'success': False})

def _ocr_work(img, bboxes, lang):
    crops = [crop_image(img, bbox) for bbox in bboxes]
    return ocr_fusion_model.best_ocr(crops, lang=lang)

@app.post('/ocr')
async def ocr(request: Request):
    img = None
    try:
        data = await request.json()
        if not data:
            return JSONResponse(content={"error": "No JSON data provided"}, status_code=400)

        image_data = data.get('image')
        is_valid, message = await run_in_threadpool(validate_image_data, image_data)
        if not is_valid:
            return JSONResponse(content={"error": f"Invalid image data: {message}"}, status_code=400)

        bboxes = data.get('bboxes', [])
        lang = data.get('lang', 'jpn')

        if not bboxes:
            return JSONResponse(content={"results": []})

        img = await run_in_threadpool(base64_to_cv2, image_data)
        results = await _gpu_stage(request, _ocr_work, img, bboxes, lang)
        return JSONResponse(content={"results": results})

    except _ClientGone:
        return JSONResponse(content={"error": "client disconnected"}, status_code=499)
    except Exception as e:
        print(f"OCR error: {str(e)}")
        return JSONResponse(content={"error": f"Processing failed: {str(e)}"}, status_code=400)

    finally:
        if img is not None:
            free_memory(img)

@app.post('/crop_faces')
async def crop_faces(request: Request):
    data = await request.json()
    img = base64_to_cv2(data['image'])
    
    # Get face bounding boxes from YOLO face model
    face_bboxes = detect_faces(img)
    
    # Proceed to crop as before with these bbox pixel coordinates
    faces = []
    img_height, img_width = img.shape[:2]
    for bbox in face_bboxes:
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(img_width, int(bbox[2]))
        y2 = min(img_height, int(bbox[3]))
        face_crop = img[y1:y2, x1:x2]
        _, buffer = cv2.imencode('.png', face_crop)
        face_base64 = base64.b64encode(buffer).decode('utf-8')
        faces.append(f"data:image/png;base64,{face_base64}")
    
    return JSONResponse(content={"faces": faces})

@app.post('/process')
async def process_image(request: Request):
    data = await request.json()
    img = base64_to_cv2(data['image'])
    img_height, img_width = img.shape[:2]
    
    # 1. Run detection
    detections = run_detection(img)
    
    def convert_box(bbox):
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2 / img_width
        center_y = (y1 + y2) / 2 / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        return [center_x, center_y, width, height]

    # 2. Prepare features
    text_feats = [convert_box(item["bbox"]) for item in detections.get("texts", [])]
    char_feats = [convert_box(item["bbox"]) for item in detections.get("characters", [])]

    if len(text_feats) == 0 or len(char_feats) == 0:
        # No texts or no characters detected — return empty or default response
        return JSONResponse(content={
            "transcript": "",
            "associations": {},
            "clusters": {},
            "speakerMap": {}
        })
    
    text_tensor = torch.tensor(text_feats, dtype=torch.float32)
    char_tensor = torch.tensor(char_feats, dtype=torch.float32)

    with torch.no_grad():
        text_embeddings, char_embeddings = EMBED_MODEL(text_tensor, char_tensor)

    text_embeddings_np = text_embeddings.cpu().numpy()
    char_embeddings_np = char_embeddings.cpu().numpy()

    cluster_map = cluster_characters(char_embeddings_np)

    relations = predict_relations(
        texts=detections["texts"],
        characters=detections["characters"],
        width=img_width,
        height=img_height,
        classifier_model=classifier_model,
        text_feats=text_feats,
        char_feats=char_feats
    )

    associations = associate_text_to_characters(
        detections["texts"],
        detections["characters"],
        relations
    )

    text_crops = [crop_image(img, item["bbox"]) for item in detections["texts"]]
    ocr_results = ocr_fusion_model.best_ocr(text_crops)

    transcript = generate_transcript(
        ocr_results,
        associations,
        cluster_map,
        character_mapper=CharacterMapper()
    )

    character_mapper2 = CharacterMapper()

    # Assuming all clusters are 'face' type for now; adjust if you have different cluster types
    speaker_map = {}
    for cluster_id in set(cluster_map.values()):
        speaker_label = character_mapper2.get_speaker("face", cluster_id)  # or "body" if applicable
        speaker_map[cluster_id] = speaker_label

    return JSONResponse(content={
        "transcript": transcript,
        "associations": associations,
        "clusters": cluster_map,
        "speakerMap": speaker_map
    })


# ✅ FAST BATCH ENDPOINT - FRONTEND SPEED UP 5-10x
def _all_data_gpu_work(img, lang):
    """Detection + OCR for one page: the part that must not run concurrently."""
    detections = run_detection(img)
    bubble_bboxes = [b["bbox"] for b in detections["bubbles"]]
    ocr_results = []
    if bubble_bboxes:
        crops = [crop_image(img, bbox) for bbox in bubble_bboxes]
        ocr_results = ocr_fusion_model.best_ocr(crops, lang=lang)
    return detections, ocr_results

@app.post('/get_all_data')
async def get_all_data(request: Request):
    """Everything the overlay needs for one page in ONE request: detections + OCR + embeddings.
    Optional body fields: "lang" (default "jpn")."""
    img = None
    try:
        data = await request.json()
        if not data:
            return JSONResponse(content={"error": "No JSON data provided"}, status_code=400)

        image_data = data.get('image')
        is_valid, message = await run_in_threadpool(validate_image_data, image_data)
        if not is_valid:
            return JSONResponse(content={"error": f"Invalid image data: {message}"}, status_code=400)

        lang = data.get('lang', 'jpn')

        img = await run_in_threadpool(base64_to_cv2, image_data)
        img_height, img_width = img.shape[:2]

        if img_height < 10 or img_width < 10:
            return JSONResponse(content={"error": "Image too small"}, status_code=400)

        detections, ocr_results = await _gpu_stage(request, _all_data_gpu_work, img, lang)

        # Embeddings are tiny geometry features: no GPU lock needed
        emb = await run_in_threadpool(
            _compute_embeddings,
            [b["bbox"] for b in detections["bubbles"]],
            [c["bbox"] for c in detections["characters"]],
            img_width, img_height
        )

        return JSONResponse(content={
            "detections": detections,
            "ocr_results": ocr_results,
            "text_embeddings": emb["text_embeddings"],
            "char_embeddings": emb["char_embeddings"],
            "timestamp": datetime.now().timestamp()
        })

    except _ClientGone:
        return JSONResponse(content={"error": "client disconnected"}, status_code=499)
    except Exception as e:
        print(f"Batch error: {str(e)}")
        return JSONResponse(content={"error": f"Processing failed: {str(e)}"}, status_code=400)

    finally:
        if img is not None:
            free_memory(img)


# ✅ PERFORMANCE MONITORING
class PerformanceMonitor:
    """Track server performance metrics"""
    def __init__(self):
        self.metrics = {
            'total_requests': 0,
            'avg_response_time': 0,
            'slow_requests': 0,
            'errors': 0
        }
        self.response_times = []
    
    def record_request(self, endpoint, duration, success=True):
        self.metrics['total_requests'] += 1
        self.response_times.append(duration)
        
        if not success:
            self.metrics['errors'] += 1
        
        if duration > 2.0:  # Slow request > 2s
            self.metrics['slow_requests'] += 1
        
        # Keep only last 100 response times
        if len(self.response_times) > 100:
            self.response_times.pop(0)
        
        # Update average
        self.metrics['avg_response_time'] = sum(self.response_times) / len(self.response_times)
    
    def get_stats(self):
        return {
            **self.metrics,
            'max_response_time': max(self.response_times) if self.response_times else 0,
            'min_response_time': min(self.response_times) if self.response_times else 0
        }

perf_monitor = PerformanceMonitor()

# ✅ PARALLEL PROCESSING EXECUTOR
executor = ThreadPoolExecutor(max_workers=4)

# ✅ PERFORMANCE MIDDLEWARE
@app.middleware("http")
async def performance_middleware(request: Request, call_next):
    start_time = time.time()
    
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        perf_monitor.record_request(request.url.path, duration, success=True)
        return response
    except Exception as e:
        duration = time.time() - start_time
        perf_monitor.record_request(request.url.path, duration, success=False)
        raise

# ✅ PERFORMANCE STATS ENDPOINT
@app.get('/performance_stats')
async def performance_stats():
    """Get server performance statistics"""
    return JSONResponse(content=perf_monitor.get_stats())

# ✅ PARALLEL IMAGE PROCESSING
async def process_images_parallel(images, max_workers=3):
    """Process multiple images concurrently"""
    loop = asyncio.get_event_loop()
    
    # Process images in parallel using thread pool
    tasks = []
    for img_data in images:
        task = loop.run_in_executor(executor, process_single_image, img_data)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results

def process_single_image(img_data):
    """Process a single image (runs in thread pool)"""
    try:
        img = base64_to_cv2(img_data)
        detections = run_detection(img)
        return {"success": True, "detections": detections}
    except Exception as e:
        return {"success": False, "error": str(e)}

# ✅ BATCH PROCESSING ENDPOINT WITH PARALLEL EXECUTION
@app.post('/detect_batch')
async def detect_batch(request: Request):
    """Detect objects in multiple images in parallel"""
    try:
        data = await request.json()
        images = data.get('images', [])
        
        if not images:
            return JSONResponse(content={"error": "No images provided"}, status_code=400)
        
        # Process in parallel (max 3 at a time to avoid OOM)
        results = await process_images_parallel(images, max_workers=3)
        
        return JSONResponse(content={
            "results": results,
            "total": len(results),
            "successful": sum(1 for r in results if isinstance(r, dict) and r.get('success'))
        })
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get('/health')
async def health():
    """Health check endpoint"""
    return JSONResponse(content={
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "performance": perf_monitor.get_stats()
    })

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        "magi_server:app",
        host="0.0.0.0",
        port=5000,
        workers=1,
        limit_concurrency=16,
        timeout_keep_alive=120,
        reload=False
    )