import os
import easyocr
import cv2
import numpy as np
from typing import List
import re
import torch
from PIL import Image
from paddleocr import PaddleOCR
from manga_ocr import MangaOcr
from transformers import DonutProcessor, VisionEncoderDecoderModel
from paddlenlp import Taskflow
import pytesseract
import warnings

# Windows users: set TESSERACT_CMD in .env (e.g. E:\tesseract\tesseract.exe)
# Linux/macOS users: leave unset — pytesseract finds it on PATH.
_tess = os.getenv('TESSERACT_CMD')
if _tess:
    pytesseract.pytesseract.tesseract_cmd = _tess


# ----------------------
# Donut OCR Wrapper
# ----------------------
class DonutOCR:
    def __init__(self, model_name="naver-clova-ix/donut-base-finetuned-cord-v2", device=None, fp16=True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fp16 = fp16 and self.device == "cuda"
        self.processor = DonutProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)

        if self.fp16:
            self.model = self.model.half()
        self.model.to(self.device)
        self.model.eval()

    def predict(self, images: List[np.ndarray], lang: str = "jpn") -> List[str]:
        rgb_images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
        results = []

        try:
            inputs = self.processor(images=rgb_images, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.half().to(self.device) if k == "pixel_values" else v.to(self.device) 
                      for k, v in inputs.items()}
            decoder_input_ids = self.processor.tokenizer("<s>", add_special_tokens=False, return_tensors="pt").input_ids.to(self.device).repeat(len(rgb_images), 1)

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["pixel_values"],
                    max_length=128,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True,
                    early_stopping=True,
                    decoder_input_ids=decoder_input_ids
                )

            results = self.processor.batch_decode(outputs, skip_special_tokens=True)
        except Exception as e:
            warnings.warn(f"DonutOCR failed: {e}")
            results = [""] * len(images)

        return results


# ----------------------
# Language Mapping for PaddleOCR
# ----------------------
lang_map = {
    "eng": "en", "spa": "es", "por": "pt", "deu": "german",
    "ind": "en", "kor": "korean", "chi_sim": "ch", "jpn": "japan", "russian": "ru"
}


# ----------------------
# Semantic Embedder
# ----------------------
class SemanticEmbedder:
    def __init__(self):
        try:
            self.embed_task = Taskflow("sentence_embedding")
        except Exception as e:
            warnings.warn(f"Failed to load PaddleNLP sentence_embedding: {e}")
            self.embed_task = None

    def get_embeddings(self, sentences: List[str]) -> np.ndarray:
        if not self.embed_task or not sentences:
            return np.zeros((len(sentences), 384))
        try:
            vectors = self.embed_task(sentences)
            if not vectors or not isinstance(vectors, list) or not all(isinstance(v, list) for v in vectors):
                return np.zeros((len(sentences), 384))
            return np.array(vectors)
        except Exception:
            return np.zeros((len(sentences), 384))

    def cosine_similarity(self, v1, v2):
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        return np.dot(v1_norm, v2_norm)

    def select_best(self, sentences: List[str]) -> str:
        if not sentences:
            return ""
        if len(sentences) == 1:
            return sentences[0]

        vectors = self.get_embeddings(sentences)
        scores = [
            np.mean([self.cosine_similarity(vectors[i], vectors[j]) for j in range(len(vectors)) if i != j])
            for i in range(len(vectors))
        ]
        best_idx = int(np.argmax(scores))
        return sentences[best_idx]


# ----------------------
# OCR Fusion with Intelligent Routing
# ----------------------
class OCRFusion:
    def __init__(self, use_donut=False):
        self.donut = DonutOCR() if use_donut else None
        self.manga = MangaOcr()
        self.paddle_cache = {}
        self.embedder = SemanticEmbedder()
        self.easyocr_reader_ru = easyocr.Reader(['ru', 'en'])

    def get_paddle(self, lang: str):
        paddle_lang = lang_map.get(lang, "en")
        if paddle_lang not in self.paddle_cache:
            self.paddle_cache[paddle_lang] = PaddleOCR(use_angle_cls=True, lang=paddle_lang)
        return self.paddle_cache[paddle_lang]

    def best_ocr(self, images: List[np.ndarray], lang: str = "eng") -> List[str]:
        final_results = []
        lang_lower = lang.lower()

        for img in images:
            candidates = []

            # --- Japanese: MangaOCR + Tesseract ---
            if lang_lower in ("jpn", "ja", "japanese"):
                try:
                    candidates.append(self.manga(Image.fromarray(img)))
                except Exception as e:
                    warnings.warn(f"MangaOCR error: {e}")
                try:
                    pil_img = Image.fromarray(img)
                    candidates.append(pytesseract.image_to_string(
                        pil_img, lang="jpn+jpan-vert", config="--oem 3 --psm 6"))
                except Exception as e:
                    warnings.warn(f"TesseractOCR (jpn) error: {e}")

            # --- Russian: EasyOCR only ---
            elif lang_lower in ("ru", "russian"):
                
                try:
                    result = self.easyocr_reader_ru.readtext(img)
                    text = " ".join([line[1] for line in result if len(line) > 1])
                    candidates.append(text)
                except Exception as e:
                    warnings.warn(f"EasyOCR (RU) error: {e}")
                try:
                    pil_img = Image.fromarray(img)
                    tesseract_text = pytesseract.image_to_string(pil_img, lang="rus+eng", config="--oem 3 --psm 6")
                    candidates.append(tesseract_text)
                except Exception as e:
                    warnings.warn(f"TesseractOCR error: {e}")
            # --- Other languages: Paddle + Tesseract ---
            else:
                try:
                    paddle = self.get_paddle(lang)
                    result = paddle.ocr(img, cls=True)
                    if result and isinstance(result, list):
                        lines = result[0] if isinstance(result[0], list) else []
                        paddle_text = " ".join([line[1][0] for line in lines if len(line) > 1])
                        candidates.append(paddle_text)
                except Exception as e:
                    warnings.warn(f"PaddleOCR error: {e}")
                try:
                    pil_img = Image.fromarray(img)
                    tesseract_text = pytesseract.image_to_string(pil_img, lang=lang, config="--oem 3 --psm 6")
                    candidates.append(tesseract_text)
                except Exception as e:
                    warnings.warn(f"TesseractOCR error: {e}")

            # --- DonutOCR (optional, fallback for all langs) ---
            if self.donut:
                try:
                    donut_text = self.donut.predict([img])[0]
                    if donut_text.strip():
                        candidates.append(donut_text)
                except Exception as e:
                    warnings.warn(f"DonutOCR error: {e}")

            # Clean and select best OCR result
            cleaned = [clean_ocr_text(c) for c in candidates if c.strip()]
            best = self.embedder.select_best(cleaned)
            final_results.append(best)

        return final_results


# ----------------------
# Utility Functions
# ----------------------
def crop_image(image: np.ndarray, bbox: List[float]) -> np.ndarray:
    h, w = image.shape[:2]
    if all(0 <= v <= 1 for v in bbox):
        bbox = [int(bbox[0]*w), int(bbox[1]*h), int(bbox[2]*w), int(bbox[3]*h)]
    x1, y1, x2, y2 = bbox
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((32, 32, 3), dtype=np.uint8)
    crop = image[y1:y2, x1:x2]
    return crop if crop.size > 0 else np.zeros((32, 32, 3), dtype=np.uint8)

def preprocess_manga_text(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    resized = cv2.resize(cleaned, (128, 32))
    return resized

def clean_ocr_text(text: str) -> str:
    text = re.sub(r'\*\*.+?\*\*', '', text)
    text = re.sub(r'[\u3000-\u303F]', ' ', text)
    text = re.sub(r'P\d+', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def batch_ocr(images: List[np.ndarray], fusion_model: OCRFusion, lang: str = "eng") -> List[str]:
    return fusion_model.best_ocr(images, lang=lang)
