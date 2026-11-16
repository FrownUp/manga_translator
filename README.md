# manga_translator_hub

Manga translation pipeline: detection → OCR → relation prediction →
context-aware translation → inpainting/overlay.

- **Backend:** FastAPI (`magi_server.py`)
- **Models:** YOLOv8 (bubble/panel/face) + MLP relation/embedding + OCR fusion
- **Frontend:** browser overlay + Selenium capture (see companion frontend repo)

---

## 1. Requirements

| Item | Version |
|---|---|
| Python | **3.12.10** (tested) |
| OS | Windows 10/11 (Linux works — see §7) |
| CUDA | 12.8 (for the GPU wheels in `requirements.txt`) |
| Chrome / Chromium | latest, for `image_capture.py` |
| ~VRAM | 6 GB+ recommended |

Everything else is pinned in `requirements.txt`.

---

## 2. Install

### 2.1 Create & activate a virtual environment

**Windows**
```bat
python -m venv venv
venv\Scripts\activate