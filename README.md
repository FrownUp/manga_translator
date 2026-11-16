# Manga Translator Hub

An offline manga translation pipeline. Point it at a manga page, get English
translations overlaid on top of the original art.

**Everything runs locally** — no cloud, no API keys, no accounts.

```
Manga page in browser
        │
        ▼
Tampermonkey userscript (widget)
        │
        ├──► server4.js:3000      CORS proxy (image fetch)
        └──► magi_server.py:5000  FastAPI backend
                ├─ YOLOv8 detection (bubbles/panels/faces)
                ├─ OCR fusion (5 engines)
                ├─ translation (context-aware)
                └─ inpainting / overlay
```

---

## 1. What you need

| Requirement | Version | Notes |
|---|---|---|
| **Python** | 3.10 – 3.12 | 3.12 recommended |
| **Node.js** | 18+ | Runs the CORS proxy |
| **NVIDIA GPU** | 6 GB VRAM min | Optional — CPU works, just slower |
| **Chrome / Chromium** | Latest | For the userscript and Selenium capture |
| **Tampermonkey** | Latest | [Chrome Web Store](https://chromewebstore.google.com/detail/tampermonkey/dhdgffkkebhmkfjojejmpbldmpobfkfo) |
| **Tesseract** *(optional)* | 5.x | Only needed for the Tesseract OCR backend |

**Disk space:** ~3 GB for code + Python deps, plus ~5 GB for models + HuggingFace cache.

**CUDA:** if you have an NVIDIA GPU, install CUDA 12.8 (the default `requirements.txt` targets `torch==2.10.0+cu128`).

---

## 2. Install

### 2.1 Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

You should see `(venv)` at the start of your prompt.

### 2.2 Install Python dependencies

**With NVIDIA GPU (CUDA 12.8):**

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

**CPU-only / no NVIDIA GPU:**

Edit `requirements.txt` first — replace these two lines:

```diff
- torch==2.10.0+cu128
- torchvision==0.25.0+cu128
+ torch==2.10.0
+ torchvision==0.25.0
```

Then:

```bash
pip install -r requirements.txt
```

### 2.3 Download the spaCy English model

```bash
python -m spacy download en_core_web_sm
```

This is a **separate step** — spaCy models aren't pip packages.

### 2.4 Verify

```bash
python check_deps.py
```

Expected output:

```
Python 3.12.10  @  ...

── REQUIRED ─────────────────────────────────────────────
  ✅  fastapi                      0.115.6
  ✅  uvicorn                      0.34.0
  ✅  torch                        2.10.0+cu128
  ...

── SUMMARY ──────────────────────────────────────────────
  31/32 importable
  0 required missing
```

`0 required missing` → you're ready for step 3.

---

## 3. Download the model weights

The repo ships **code only**. Model weights are ~80 MB total and live in
**GitHub Releases**, not in the repo.

### 3.1 Automatic (recommended)

```bash
python scripts/download_models.py
```

This downloads all 7 model files from the latest GitHub Release into
`models/` at the repo root.

### 3.2 Manual

1. Open the [Releases page](https://github.com/FrownUp/manga_translator_hub/releases)
2. Download the 7 assets
3. Create a `models/` folder at the repo root:
   ```bash
   mkdir models
   ```
4. Move all 7 downloaded files into `models/`

### 3.3 Required files

| File | Purpose | Size |
|---|---|---|
| `bubble_best.pt` | Bubble + character detector | ~22 MB |
| `panel_best.pt` | Panel detector | ~22 MB |
| `face_best.pt` | Face detector | ~22 MB |
| `sgg_relation_model.pth` | Scene-graph embedding | ~1 MB |
| `relation_classifier3.pth` | Relation classifier | ~50 KB |
| `face_embedding_model.pth` | Face embedding | ~5 MB |
| `manga_embedding_model2.pth` | Manga-specific embedding | ~5 MB |

### 3.4 Verify

```bash
python model_paths.py
```

Expected output — all checkmarks:

```
MODEL_DIR = /path/to/manga_translator_hub/models

  ✓ bubble_best.pt
  ✓ panel_best.pt
  ✓ face_best.pt
  ✓ sgg_relation_model.pth
  ✓ relation_classifier3.pth
  ✓ face_embedding_model.pth
  ✓ manga_embedding_model2.pth
```

Any ✗ → that file is missing or misnamed. Re-download it.

### 3.5 Folder layout after download

```
manga_translator_hub/
├── magi_server.py
├── model_paths.py
├── ...
└── models/
    ├── bubble_best.pt
    ├── panel_best.pt
    ├── face_best.pt
    ├── sgg_relation_model.pth
    ├── relation_classifier3.pth
    ├── face_embedding_model.pth
    └── manga_embedding_model2.pth
```

---

## 4. Run

Three things need to be running. **Order matters.**

### 4.1 Terminal 1 — Node.js proxy

```bash
node server4.js
```

Expected:

```
server4.js running on http://localhost:3000
```

Keep this terminal open.

### 4.2 Terminal 2 — FastAPI backend

Activate the venv first, then:

```bash
python magi_server.py
```

Expected after ~5 seconds:

```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:5000
```

Models load **lazily** — you'll see logs like `Loading model: bubble_char`
on the first request, not at startup. That's normal.

### 4.3 Browser — install the userscript

1. Click the **Tampermonkey** toolbar icon → **Dashboard**
2. Drag `userscripts/Manga_OCR_Pro_tampermonkeylight.txt` into the dashboard
   (or: **Utilities** → **Import from file**)
3. **Read the `@match` lines** on the install preview — these decide which
   sites the script runs on. Add a line for every manga site you use:
   ```javascript
   // @match   *://*.mangadex.org/*
   // @match   *://*.mangakakalot.com/*
   // @match   *://*.comick.io/*
   ```
   ⛔ **Never** use the catch-all `*://*/*` — it runs on every website including your bank.
4. Click **Install**
5. Open a manga chapter → a small widget appears in the bottom-right corner

### 4.4 Use it

| Widget button | What it does |
|---|---|
| **OCR** | Detect → OCR → translate → overlay on current viewport |
| **Re-OCR** | Force a fresh run (needed after page navigation) |
| **Hide original / Show original** | Toggle bilingual vs. inpainted view |
| **Chars** *(full version only)* | Character-dialogue panel |

**Typical flow:**

1. Open a manga chapter
2. Click **OCR** in the widget
3. First call takes 10–20 s (models warm up); after that, <1 s
4. Scroll down → click **OCR** again, or enable `AUTO_OCR` in the userscript `CONFIG`

---

## 5. Verify everything works

### 5.1 Backend

```bash
curl http://localhost:5000/system_status
```

Expect:

```json
{"backend_healthy": true, "models_loaded": true, "inpainting": true, ...}
```

### 5.2 End-to-end translation test

```bash
curl -X POST http://localhost:5000/translate ^
     -H "Content-Type: application/json" ^
     -d "{\"text\":\"こんにちは\",\"source_language\":\"auto\"}"
```

Expect:

```json
{"translated_text":"Hello", ...}
```

### 5.3 Browser

Open DevTools (`F12`) → **Network** tab → click **OCR** in the widget.
You should see successful (`200`) requests to:

- `POST http://localhost:3000/api/image-to-base64`
- `POST http://localhost:5000/detect`
- `POST http://localhost:5000/ocr`
- `POST http://localhost:5000/translate_batch`
- `POST http://localhost:5000/create_overlay`

Any `CORS` error or `ERR_CONNECTION_REFUSED` → the corresponding service
isn't running (see step 4).

---

## 6. Troubleshooting

### Install errors

| Error | Fix |
|---|---|
| `pip install` fails on `torch==2.10.0+cu128` | Missing `--extra-index-url https://download.pytorch.org/whl/cu128` (step 2.2) |
| `No module named 'en_core_web_sm'` | Run `python -m spacy download en_core_web_sm` (step 2.3) |
| `TesseractNotFoundError` at runtime | Install Tesseract binary — see step 2's requirement table |
| `pip check` reports torch conflicts | Reinstall torch from the CUDA index **after** installing OCR packages |

### Startup errors

| Error | Fix |
|---|---|
| `EADDRINUSE :3000` | Port 3000 taken — kill the process or change `PORT` in `server4.js` (and update the userscript `CONFIG.PROXY_URL`) |
| `address already in use :5000` | Port 5000 taken — `python magi_server.py --port 5001` and update the userscript `CONFIG.BACKEND_URL` |
| `CUDA out of memory` | Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before starting, or run CPU-only |
| `FileNotFoundError: bubble_best.pt` on first `/detect` | Model missing from `models/` — re-run step 3 |

### Runtime errors

| Symptom | Fix |
|---|---|
| Widget doesn't appear on a manga page | Tampermonkey script enabled? URL matches `@match`? Hard refresh (`Ctrl+Shift+R`) |
| Widget appears but **OCR** does nothing | DevTools → Network → does `:5000/detect` fire? `ERR_CONNECTION_REFUSED` = backend not running |
| Translation overlay misaligned | Browser zoom ≠ 100% — reset with `Ctrl+0` |
| OCR returns empty for every bubble | Wrong source language detected — force `jpn` or `ko` in widget settings |
| Two widgets appear | You installed both userscripts — uninstall one |

---

## 7. Project structure

```
manga_translator_hub/
├── README.md                 ← you are here
├── requirements.txt          ← Python deps
├── model_paths.py            ← model file locations
├── scripts/
│   └── download_models.py    ← fetches weights from Releases
│
├── models/                   ← weights (git-ignored, downloaded separately)
│
├── server4.js                ← Node CORS proxy (start FIRST)
├── magi_server.py            ← FastAPI backend (start SECOND)
│
├── userscripts/
│   ├── Manga_OCR_Pro_tampermonkeylight.txt   ← recommended (fast)
│   └── Manga_OCR_Pro_tampermonkey.txt        ← full UI
│
├── detection_processor.py    ← YOLO → bbox conversion
├── face_processor.py         ← face detect + embed
├── character_mapper.py       ← face → speaker identity
├── ocr_processor.py          ← OCR fusion (5 engines)
├── relation_predictor.py     ← text↔character relation scoring
├── association_engine.py     ← DBSCAN character clustering
├── transcript_generator.py   ← reading-order transcript
│
├── translator.py             ← translation + cache
├── integrated_translator.py  ← context + quality analysis
├── virtual_manga_page.py     ← reading-order virtual page
├── bubble_tracker.py         ← per-bubble lifecycle state
│
├── image_capture.py          ← Selenium capture
├── realtime_overlay.py       ← text rendering
├── manga_inpainter.py        ← opencv / SD inpainting
├── bulk_processor.py         ← batch processing
└── backend_integration.py    ← HTTP client
```

---

## 8. One-page quick start

```bash
# ── Setup (one time) ─────────────────────────────────────
python -m venv venv
venv\Scripts\activate                          # Windows
# source venv/bin/activate                     # Linux/macOS

pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
python -m spacy download en_core_web_sm

# ── Download models ──────────────────────────────────────
python scripts/download_models.py

# ── Verify ───────────────────────────────────────────────
python check_deps.py                # → "0 required missing"
python model_paths.py               # → all ✓

# ── Run (2 terminals) ────────────────────────────────────
# Terminal 1:
node server4.js                     # → :3000

# Terminal 2:
python magi_server.py               # → :5000

# ── Browser ──────────────────────────────────────────────
# 1. Install Tampermonkey extension
# 2. Import userscripts/Manga_OCR_Pro_tampermonkeylight.txt
# 3. Allow your manga sites (@match lines)
# 4. Open a manga chapter → click OCR
```

---

## License

**AGPL-3.0** — required by the upstream Ultralytics YOLOv8 license.
See [LICENSE](LICENSE) for the full text.
