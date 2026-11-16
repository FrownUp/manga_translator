# manga\_translator\_hub

Manga translation pipeline that runs **entirely on your local machine**:

```
manga site in browser
        │
        ▼
Tampermonkey userscript  (frontend widget)
        │
        ├──► server4.js:3000        Node.js CORS proxy (image fetch)
        │
        └──► magi\_server.py:5000    FastAPI backend
                ├─ YOLOv8 detection (bubbles / panels / characters / faces)
                ├─ OCR fusion (easyocr / manga-ocr / tesseract / paddle)
                ├─ relation prediction + character association
                ├─ translation (deep-translator, spaCy quality analysis)
                └─ overlay / inpainting (opencv / Stable Diffusion)
```

**No cloud. No API keys. Everything is local.**

\---

## Table of contents

1. [What you need before starting](#1-what-you-need-before-starting)
2. [Install — step by step](#2-install--step-by-step)
3. [Configure](#3-configure)
4. [Run — full startup sequence](#4-run--full-startup-sequence)
5. [Use — reading a manga chapter](#5-use--reading-a-manga-chapter)
6. [Verify everything works](#6-verify-everything-works)
7. [Project layout](#7-project-layout)
8. [Troubleshooting](#8-troubleshooting)
9. [Known bugs / gotchas](#9-known-bugs--gotchas)
10. [Advanced: reproducible installs](#10-advanced-reproducible-installs)
11. [Linux / macOS notes](#11-linux--macos-notes)
12. [One-page quick start](#12-one-page-quick-start)

\---

## 1\. What you need before starting

### 1.1 Hardware \& OS

|Item|Minimum|Recommended|
|-|-|-|
|OS|Windows 10 / Ubuntu 20.04 / macOS 12|Windows 11 / Ubuntu 22.04|
|GPU|Any NVIDIA with 6 GB VRAM (or CPU-only)|RTX 3060+ / 8 GB+ VRAM|
|RAM|8 GB|16 GB+|
|Disk|5 GB free|15 GB+ (models + cache)|

### 1.2 Software

|Software|Version|Why|
|-|-|-|
|**Python**|**3.12.x** (3.10–3.12 all work)|Backend runtime|
|**Node.js**|**18+**|Runs `server4.js` CORS proxy|
|**CUDA**|**12.8**|Required only for GPU wheels in `requirements.txt`|
|**Chrome / Chromium**|latest|Used by Tampermonkey *and* by `image\_capture.py` (Selenium)|
|**Tampermonkey**|latest|Frontend userscript host|
|**Tesseract** *(optional)*|5.x|Only needed if you want pytesseract OCR backend|
|**GTK / Cairo DLLs** *(optional)*|—|Only needed on Windows for `cairosvg` SVG path|

### 1.3 Files you should have on disk

```
manga\_translator\_hub/
├── requirements.txt          ← pinned Python deps
├── server4.js                ← Node proxy
├── magi\_server.py            ← FastAPI backend
├── model\_paths.py            ← model paths (you edit this)
├── userscripts/
│   ├── Manga\_OCR\_Pro\_tampermonkeylight.txt
│   └── Manga\_OCR\_Pro\_tampermonkey.txt
├── model files (\*.pt, \*.pth)  ← see §3.2
└── all other .py modules
```

If any of these are missing, see §7 (project layout) for the full file list.

\---

## 2\. Install — step by step

### 2.1 Clone / copy the project

```bash
cd path/to/your/project
```

(No `git clone` if you're already sitting in the folder.)

### 2.2 Create a Python virtual environment

**Windows**

```bat
python -m venv venv
venv\\Scripts\\activate
```

**Linux / macOS**

```bash
python -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the start of your prompt. If not, activation failed — check that you're pointing at the right `python`.

### 2.3 Install Python dependencies

`requirements.txt` contains **CUDA 12.8 wheels** for `torch` / `torchvision`
(`torch==2.10.0+cu128`). These are **not on PyPI** — they live on PyTorch's
own index. That's why the install command needs `--extra-index-url`:

**Windows**

```bat
pip install -r requirements.txt ^
  --extra-index-url https://download.pytorch.org/whl/cu128
```

**Linux / macOS**

```bash
pip install -r requirements.txt \\
  --extra-index-url https://download.pytorch.org/whl/cu128
```

> \*\*CPU-only machine?\*\*  
> Open `requirements.txt`, replace:
> ```
> torch==2.10.0+cu128
> torchvision==0.25.0+cu128
> ```
> with:
> ```
> torch==2.10.0
> torchvision==0.25.0
> ```
> then run plain `pip install -r requirements.txt` (no extra index needed).
> Everything else works the same; just much slower on first inference.

### 2.4 Download the spaCy English model

```bash
python -m spacy download en\_core\_web\_sm
```

This is a **separate step** from `pip install` — spaCy models aren't pip packages.

### 2.5 Verify Node.js

```bash
node --version
# should print v18.x or higher
```

If missing: https://nodejs.org/ → download LTS → install → restart terminal.

### 2.6 Verify the Python environment

```bash
python check\_deps.py
```

Expected output (abridged):

```

## 📦 Download Model Weights

This repository ships **code only**. The 7 trained model files must be
downloaded separately from the **GitHub Releases** page and placed in a
`models/` folder at the repo root.

### Why not in the repo?

Model weights are 20–50 MB each. Including them in git history would:
- Hit GitHub's 100 MB per-file limit
- Permanently bloat every `git clone`
- Burn through Git LFS bandwidth quotas

So the code lives here, and the weights live in Releases. Clone is fast (~1 MB), and models download in one command.

### Required model files

| File | Purpose | Size |
|---|---|---|
| `bubble_best.pt` | YOLO bubble + character detector | ~22 MB |
| `panel_best.pt` | YOLO manga panel detector | ~22 MB |
| `face_best.pt` | YOLO manga face detector | ~22 MB |
| `sgg_relation_model.pth` | Scene-graph embedding model | ~1 MB |
| `relation_classifier3.pth` | Text↔character relation classifier | ~50 KB |
| `face_embedding_model.pth` | Face embedding (character identification) | ~1–5 MB |
| `manga_embedding_model2.pth` | Manga-specific embedding (DASS) | ~1–5 MB |

**Total: ~70–80 MB**

### Option A — Automatic download (recommended)

After cloning the repo, run:

```bash
python scripts/download_models.py
```

The script downloads all 7 files from the latest GitHub Release into `models/`.

### Option B — Manual download

1. Open the [Releases page](https://github.com/FrownUp/manga_translator_hub/releases)
2. Download the latest release's assets (the 7 `.pt` / `.pth` files)
3. Create a `models/` folder at the repo root if it doesn't exist:
   ```bash
   mkdir models
   ```
4. Move or copy all 7 downloaded files into `models/`

### Verify the setup

```bash
python model_paths.py
```

Expected output:

```
MODEL_DIR = /path/to/manga_translator_hub/models

  ✓ bubble_best.pt  (./models/bubble_best.pt)
  ✓ panel_best.pt  (./models/panel_best.pt)
  ✓ face_best.pt  (./models/face_best.pt)
  ✓ sgg_relation_model.pth  (./models/sgg_relation_model.pth)
  ✓ relation_classifier3.pth  (./models/relation_classifier3.pth)
  ✓ face_embedding_model.pth  (./models/face_embedding_model.pth)
  ✓ manga_embedding_model2.pth  (./models/manga_embedding_model2.pth)
```

All ✓ → you're ready to run `python magi_server.py`.

### Expected folder layout after download

```
manga_translator_hub/
├── magi_server.py
├── manga_pipeline.py
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
======================================================================
Python 3.12.10  @  C:\\path\\to\\venv\\Scripts\\python.exe
======================================================================

── REQUIRED ─────────────────────────────────────────────
  ✅  fastapi                      0.115.6
  ✅  uvicorn                      0.34.0
  ✅  numpy                        2.2.0
  ✅  torch                        2.10.0+cu128
  ✅  ultralytics                  8.3.36
  ...

── SUMMARY ──────────────────────────────────────────────
  31/32 importable
  0 required missing
```

If it says `0 required missing` ✅ → you're ready for §3.

If **any** required dep is missing → run:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

### 2.7 Install the Tampermonkey browser extension

Pick your browser:

|Browser|Install link|
|-|-|
|Chrome / Edge / Brave|https://chromewebstore.google.com/detail/tampermonkey/dhdgffkkebhmkfjojejmpbldmpobfkfo|
|Firefox|https://addons.mozilla.org/firefox/addon/tampermonkey/|
|Safari|https://apps.apple.com/app/tampermonkey/id1482490089|
|Opera|https://addons.opera.com/extensions/details/tampermonkey-beta/|

After install, **pin the Tampermonkey icon** to your toolbar — you'll need it
to toggle the script per-site.

### 2.8 Install a userscript

Two scripts ship in `userscripts/` — install **one**, not both:

|File|When to use|
|-|-|
|**`Manga\_OCR\_Pro\_tampermonkeylight.txt`**|Minimal UI. Only OCRs the visible viewport. Fastest. Perfect for reading.|
|`Manga\_OCR\_Pro\_tampermonkeyheavy.txt`|Full UI. Batch OCR, per-bubble re-translate, character-dialogue panel. Heavier.|

**To install:**

1. Click the Tampermonkey toolbar icon → **Dashboard**.
2. Drag the `.txt` file from Explorer/Finder into the dashboard page
*or* → **Utilities** tab → **Import from file**.
3. Tampermonkey shows a preview page. **Read the `@match` lines** (§2.9) —
this is what decides which sites the script runs on.
4. Click **Install**.
5. Open any manga tab. A small floating widget should appear in the
bottom-right corner.

> 💡 `.txt` works fine for Tampermonkey import. If your OS blocks dropping
> `.txt` files, rename the file to `.user.js` first — Tampermonkey will
> recognize it immediately.

### 2.9 Allow sites (and block the rest)

The userscript runs **only** on URLs matched by its `@match` lines.
Restrict this to sites you actually read manga on — the script sends page
content (images) to your local backend, and you don't want it running on
your bank or email.

**During install**, the script already contains a safe allowlist. Typical
block to edit in the Tampermonkey preview screen (or afterward via
*Dashboard → script name → Edit*):

```javascript
// @match        \*://\*.mangadex.org/\*
// @match        \*://\*.mangakakalot.com/\*
// @match        \*://\*.mangahere.com/\*
// @match        \*://\*.comick.io/\*
// @match        \*://\*.mangapark.net/\*
// @match        \*://\*.batotoo.com/\*
// @match        \*://\*.mangafire.to/\*
```

Add a line for every site you use.

⛔ **Never** use the catch-all:

```javascript
// @match        \*://\*/\*       ← NO — this runs on EVERY website
```

…unless you're testing and will disable it right after.

**Extra safety (recommended on shared machines):**

1. Dashboard → script → **Settings** tab
2. **User excludes** — add `\*://\*.bank.com/\*`, `\*://mail.google.com/\*`, etc.
3. **Run at** → `document-idle` (default — don't use `document-start`)
4. **Inject into** → `content` (not `page`)

### 2.10 Configure the userscript

The script assumes these defaults — change only if you moved ports:

```javascript
const CONFIG = {
  BACKEND\_URL:  "http://localhost:5000",
  PROXY\_URL:    "http://localhost:3000/api/image-to-base64",
  AUTO\_OCR:     false,        // true → OCR every page automatically
  SHOW\_ORIGINAL: true,        // toggle original + translation vs. inpainted
  TARGET\_LANG:  "en",
};
```

Save (`Ctrl+S`), reload the manga tab.

**Done with install.** → §3 to configure, then §4 to run.

\---

## 3\. Configure

### 3.1 Environment variables (recommended)

Set these before starting `magi\_server.py`.

**Windows (cmd)**

```bat
set HF\_HOME=E:\\py\\hf\_cache
set TRANSFORMERS\_NO\_TF=1
set PYTORCH\_CUDA\_ALLOC\_CONF=expandable\_segments:True
```

**Windows (PowerShell)**

```powershell
$env:HF\_HOME="E:\\py\\hf\_cache"
$env:TRANSFORMERS\_NO\_TF="1"
$env:PYTORCH\_CUDA\_ALLOC\_CONF="expandable\_segments:True"
```

**Linux / macOS**

```bash
export HF\_HOME=$HOME/.cache/huggingface
export TRANSFORMERS\_NO\_TF=1
export PYTORCH\_CUDA\_ALLOC\_CONF=expandable\_segments:True
```

|Variable|Why|
|-|-|
|`HF\_HOME`|Where Hugging Face models are cached. Silences a `TRANSFORMERS\_CACHE` deprecation warning.|
|`TRANSFORMERS\_NO\_TF=1`|Stops `transformers` from loading TensorFlow. Removes the `tf.losses.sparse\_softmax\_cross\_entropy` deprecation spam. You don't need TF — this project is PyTorch-only.|
|`PYTORCH\_CUDA\_ALLOC\_CONF=expandable\_segments:True`|Reduces VRAM fragmentation across many YOLO/OCR calls. `magi\_server.py` also sets this internally, but having it exported avoids the very first allocation failing on some drivers.|

### 3.2 Model paths — **edit `model\_paths.py`**

Open `model\_paths.py` and point each path at the actual model on your disk:

```python
MODEL\_DIR               = r"E:\\magidataset"

YOLO\_BUBBLE\_MODEL\_PATH  = rf"{MODEL\_DIR}\\bubble\_detect\\content\\runs\\detect\\train\\weights\\best.pt"
YOLO\_PANEL\_MODEL\_PATH   = r"E:\\py\\panel model\\best.pt"
YOLO\_FACE\_MODEL\_PATH    = rf"{MODEL\_DIR}\\yolo\_Face\\runs\\runs\\detect\\yolov8\_manga\_faces\\weights\\best.pt"

EMBED\_MODEL\_PATH        = rf"{MODEL\_DIR}\\magi\_sgg\_relation\_model.pth"
CLASSIFIER\_MODEL\_PATH   = rf"{MODEL\_DIR}\\relation\_classifier.pth"     # ← your path
```

|Model|What it does|Typical size|
|-|-|-|
|`YOLO\_BUBBLE\_MODEL\_PATH`|Detects speech bubbles + character bodies|\~25 MB|
|`YOLO\_PANEL\_MODEL\_PATH`|Detects panel borders (for reading order)|\~25 MB|
|`YOLO\_FACE\_MODEL\_PATH`|Detects faces (for character identity)|\~25 MB|
|`EMBED\_MODEL\_PATH`|Maps text/char boxes → 32-d embedding vectors|\~50 KB|
|`CLASSIFIER\_MODEL\_PATH`|MLP that scores text↔character relations|\~10 KB|

If a path is wrong, the server **still starts** — models load **lazily** on
first request. You'll see `FileNotFoundError` only when you hit `/detect`.
Fix the path in `model\_paths.py` and restart.

### 3.3 Ports (only if you need to change them)

|Service|Default|Where to change|
|-|-|-|
|Node proxy|`3000`|Top of `server4.js` → `const PORT = 3000;`|
|FastAPI backend|`5000`|`uvicorn magi\_server:app --port 5000`|
|Userscript targets|`3000` + `5000`|§2.10 — edit `CONFIG` in the userscript|

**If you change any of these, change both sides** (backend + userscript
`CONFIG`, and/or `server4.js` + userscript proxy URL) — otherwise the
widget will silently fail with CORS errors.

\---

## 4\. Run — full startup sequence

Order matters: **Node proxy → FastAPI backend → browser**.

### 4.1 Terminal 1 — Node.js proxy

```bash
node server4.js
```

Expected output:

```
server4.js running on http://localhost:3000
```

Keep this terminal open. If it crashes with `EADDRINUSE`:

* Another process is on port 3000 → kill it (`netstat -ano | findstr :3000`
on Windows, `lsof -i :3000` on Linux), or change `PORT` in `server4.js`
and update the userscript `CONFIG.PROXY\_URL`.

> ⚠️ \*\*Do not expose port 3000 to the internet.\*\* It has no auth and will
> fetch \*\*any\*\* URL it's given. Localhost only.

### 4.2 Terminal 2 — FastAPI backend

Activate the venv first (`venv\\Scripts\\activate` / `source venv/bin/activate`), then:

```bash
python magi\_server.py
```

Expected output (after \~5 seconds):

```
INFO:     Started server process \[...]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:5000 (Press CTRL+C to quit)
```

Keep this terminal open. On first request, you'll see model-loading logs
(`Loading model: bubble\_char`, etc.) — that's normal, models are lazy.

Alternative (bypasses `magi\_server.py`'s entry point):

```bash
uvicorn magi\_server:app --host 127.0.0.1 --port 5000
```

> ⚠️ Do \*\*not\*\* use `--host 0.0.0.0` unless you intend LAN access. The
> backend has no auth — anyone on your network could hit it.

### 4.3 Terminal 3 (browser) — open a manga site

1. Open any manga chapter on a site from your `@match` allowlist.
2. Confirm the Tampermonkey widget appears (bottom-right corner).
3. Click the widget → **OCR** button.
4. First call takes \~10–20 s (models warm up). Subsequent calls are fast.
5. Translation overlays appear on top of the original bubbles.

**Done.** See §5 for widget details and §6 for verification.

\---

## 5\. Use — reading a manga chapter

### 5.1 Widget buttons

|Button|What it does|
|-|-|
|**OCR**|Detect → OCR → translate → overlay on current viewport|
|**Re-OCR**|Force a fresh run (needed after page navigation)|
|**Hide original / Show original**|Toggle between inpainted and bilingual overlay|
|**Translate bubble** *(full version only)*|Click a single bubble to re-translate it with context|
|**Chars** *(full version only)*|Opens character-dialogue panel (uses `bubble\_tracker`)|

### 5.2 Typical flow per page

```
1. Page loads
2. Click widget → OCR
3. Translation overlays on top of bubbles
4. Scroll down → widget follows → click OCR again
   (or enable AUTO\_OCR in CONFIG — slower on big pages)
```

### 5.3 State is persistent

Bubble translations are cached by `bubble\_tracker.py` on the backend. If
you re-OCR the same page, previously-translated bubbles come back
instantly — no re-translation cost.

If you want to **clear** the cache (e.g. after fixing a bug):

```bash
curl -X POST http://localhost:5000/clear\_translation\_cache
```

### 5.4 Speed tips

* **`light` userscript** (recommended) — only OCRs the visible viewport.
* **Reduce batch size** in `magi\_server.py` if VRAM is tight.
* **Set `AUTO\_OCR: false`** in `CONFIG` — manual OCR is faster because it
doesn't wait for images to fully load.
* **Reset browser zoom to 100%** — the overlay position calculation
assumes unscaled CSS pixels.

\---

## 6\. Verify everything works

### 6.1 Backend health

```bash
curl http://localhost:5000/system\_status
```

Expected:

```json
{"inpainting": true, "bulk\_processing": true, "realtime\_overlay": true, "backend\_healthy": true, "models\_loaded": true, ...}
```

If `backend\_healthy` is `false` → check Terminal 2 (backend) for exceptions.

### 6.2 Node proxy health

```bash
curl http://localhost:3000/
```

Expected: an HTTP response (200 or 404). A 404 means the server is
running but the root path isn't routed — that's fine, `/api/image-to-base64`
is the real endpoint.

### 6.3 Full end-to-end test

With all three running:

```bash
# 1. Does Node proxy reach the backend?
curl -X POST http://localhost:3000/api/image-to-base64 ^
     -H "Content-Type: application/json" ^
     -d "{\\"imageUrl\\":\\"https://example.com/test.jpg\\"}"
```

Should return `{"base64": "..."}`.

```bash
# 2. Does the backend translate?
curl -X POST http://localhost:5000/translate ^
     -H "Content-Type: application/json" ^
     -d "{\\"text\\":\\"こんにちは\\",\\"source\_language\\":\\"auto\\"}"
```

Should return `{"original\_text":"こんにちは","translated\_text":"Hello",...}`.

```bash
# 3. Can you OCR?
curl -X POST http://localhost:5000/ocr ^
     -H "Content-Type: application/json" ^
     -d "{\\"image\\":\\"<base64...>\\",\\"bboxes\\":\[]}"
```

(Will return `{"results": \[]}` for empty bboxes — that's a pass.)

### 6.4 Browser verification

Open DevTools (`F12`) → **Network** tab → click **OCR** in the widget:

|Request|Expect|
|-|-|
|`POST http://localhost:3000/api/image-to-base64`|`200`, JSON body|
|`POST http://localhost:5000/detect`|`200`, detection JSON|
|`POST http://localhost:5000/ocr`|`200`, OCR results|
|`POST http://localhost:5000/translate\_batch`|`200`, translations|
|`POST http://localhost:5000/create\_overlay`|`200`, base64 image|

Any `CORS` or `net::ERR\_CONNECTION\_REFUSED` → the corresponding service
isn't running (check §4).

\---

## 7\. Project layout

```
manga\_translator\_hub/
│
├── README.md                     ← you are here
├── requirements.txt              ← pinned Python deps
├── requirements.lock.txt         ← (optional) pip freeze snapshot
├── check\_deps.py                 ← dependency scanner
├── directory\_map.py              ← prints this tree
├── find\_missing.py               ← local-module import checker
├── model\_paths.py                ← model paths — EDIT THIS (§3.2)
│
├── server4.js                    ← Node.js CORS proxy (start FIRST, §4.1)
│
├── userscripts/                  ← Tampermonkey frontend (§2.8)
│   ├── Manga\_OCR\_Pro\_tampermonkeylight.txt   ← recommended
│   └── Manga\_OCR\_Pro\_tampermonkey.txt        ← full UI
│
├── magi\_server.py                ← FastAPI app (start SECOND, §4.2)
│
├── detection\_processor.py        ← YOLO → MAGI bbox conversion, panel sorting
├── face\_processor.py             ← face detect + embed (currently unused)
├── character\_mapper.py           ← cluster / persist face↔speaker identity
├── ocr\_processor.py              ← OCRFusion: easyocr / manga-ocr / tess / paddle
├── relation\_predictor.py         ← text↔character geometric features
├── association\_engine.py         ← DBSCAN char clustering + text association
├── transcript\_generator.py       ← reading-order transcript
│
├── translator.py                 ← deep-translator + cache + language detect
├── integrated\_translator.py      ← spaCy quality + context analysis
├── virtual\_manga\_page.py         ← virtual page + reading order
├── bubble\_tracker.py             ← per-bubble lifecycle state
│
├── image\_capture.py              ← Selenium capture (URL / canvas / screenshot)
├── realtime\_overlay.py           ← PIL / OpenCV text compositing
├── manga\_inpainter.py            ← opencv / SD / (optional LaMa) inpainting
├── bulk\_processor.py             ← batch pipeline + zip output
└── backend\_integration.py        ← HTTP client for the above endpoints
```

**Model files** (not in repo, path-controlled by `model\_paths.py`):

```
E:\\magidataset\\
├── bubble\_detect\\...\\best.pt
├── yolo\_Face\\...\\best.pt
├── magi\_sgg\_relation\_model.pth
└── relation\_classifier.pth

E:\\py\\panel model\\best.pt
```

**Runtime-created directories** (safe to delete):

```
batch\_output/         ← /process\_batch output
ocr\_data\_set/         ← OCR training data (if collected)
debug/                ← debug image dumps
```

\---

## 8\. Troubleshooting

### 8.1 Install-time errors

|Error|Fix|
|-|-|
|`pip install` fails on `torch==2.10.0+cu128`|Missing `--extra-index-url https://download.pytorch.org/whl/cu128` (§2.3)|
|`No module named 'en\_core\_web\_sm'` after install|Run `python -m spacy download en\_core\_web\_sm` (§2.4)|
|`pip install` fails on `lama-cleaner`|**Don't install it** in this venv (§9.1)|
|`TesseractNotFoundError` at runtime|Install Tesseract binary (§9.3)|
|`\[WARNING] pkg\_resources is deprecated`|`pip install "setuptools<81"` (§9.6)|
|`pip check` reports torch conflicts|One of the OCR backends (`paddleocr`, `manga-ocr`) pinned an older torch — reinstall torch from the CUDA index after installing them|

### 8.2 Startup errors

|Error|Fix|
|-|-|
|`node: command not found`|Install Node.js 18+ and reopen terminal|
|`Error: listen EADDRINUSE :3000`|Another process owns port 3000 — kill it or change `PORT` in `server4.js`|
|`\[Errno 10048] address already in use` (Python)|Port 5000 taken — `uvicorn magi\_server:app --port 5001` + update userscript `CONFIG`|
|`AttributeError: 'FastAPI' object has no attribute 'logger'`|Known bug (§9.7) — patches in file|
|`CUDA out of memory` on first request|Set `PYTORCH\_CUDA\_ALLOC\_CONF=expandable\_segments:True` (§3.1); or run CPU-only|
|`FileNotFoundError: best.pt` on first `/detect`|Wrong path in `model\_paths.py` (§3.2)|

### 8.3 Runtime errors

|Error|Fix|
|-|-|
|`Detected that datasets module was imported before paddlenlp`|Reorder imports in `ocr\_processor.py` (§9.4) — cosmetic but may break paddle OCR|
|`Could not find files for the given pattern(s)` (INFO)|Harmless `webdriver-manager` probe on Windows — ignore|
|`chrome driver` download fails|Set a proxy for webdriver-manager, or install ChromeDriver manually and set `CHROME\_DRIVER\_PATH`|
|`/capture\_image` returns 400 on headless server|Chrome needs a display — use `--headless=new` (already set) or wrap in `xvfb-run` on Linux|
|`cv2.error: OpenCV(...)` on some images|PNG with alpha channel — `base64\_to\_cv2()` handles RGBA, but exotic formats may need `cv2.IMREAD\_UNCHANGED`|
|`NameError: name 'json' is not defined` in `/process\_batch`|Known bug (§9.7) — add `import json` to `bulk\_processor.py`|

### 8.4 Browser / widget errors

|Symptom|Fix|
|-|-|
|Widget doesn't appear on a manga page|(a) Tampermonkey script enabled in toolbar? (b) URL matches `@match`? (c) `Ctrl+Shift+R` hard refresh|
|Widget appears but **OCR** does nothing|DevTools → Network → does `:5000/detect` fire? If `net::ERR\_CONNECTION\_REFUSED` → backend not running (§4.2)|
|`:3000` requests fail but `:5000` works|`server4.js` not running (§4.1)|
|CORS error in console|Backend not running, or you changed `allow\_origins` in `magi\_server.py` (default is `\*`, safe for local)|
|OCR returns empty for every bubble|Wrong source language detected — open widget settings, force `jpn` or `ko`|
|Translation overlay misaligned|Browser zoom ≠ 100% — reset with `Ctrl+0`|
|Two widgets appear|Both userscripts installed — uninstall one (§2.8)|
|Widget works on site A but not B|Site B uses a `<canvas>` the script can't read directly. Ensure `server4.js` is running and site B's domain is allowed in it|
|`AUTO\_OCR` fires before images load|Disable `AUTO\_OCR` in `CONFIG` (§2.10), click manually|

### 8.5 Data/quality errors

|Symptom|Fix|
|-|-|
|Translations are garbage|Wrong source language — force it in widget settings|
|Character names translated (`Naruto` → `Naruto-san`)|Expected — Japanese honorifics are preserved. Use full userscript for per-bubble override|
|Same bubble re-translated on every OCR|Bug — `bubble\_tracker` cache miss. Check `/translation\_stats`|
|`character\_dialogue` returns empty|No faces detected in the current page — the face model may need a higher-res image|

\---

## 9\. Known bugs / gotchas

These are **application-level** issues visible in the shipped source files.
They don't block install/run, but they'll bite you at runtime if you don't
patch them.

### 9.1 `lama-cleaner` — do not install

`manga\_inpainter.py` uses `opencv` or Stable Diffusion (via `diffusers` +
`accelerate`, both installed) as inpainting backends.

⛔ **Do not** `pip install lama-cleaner` in this venv. It pins `torch<2.5`
and will silently **downgrade** your CUDA 12.8 torch build, breaking YOLO
and inference performance.

If you truly need LaMa:

* Create a **separate venv** for it
* Call it via subprocess from `manga\_inpainter.py`
* Or skip — `opencv` + SD cover most use cases

### 9.2 Two language detectors

Both `lingua-language-detector` and `langdetect` are installed. `translator.py`
uses one. To see which:

```bash
# Windows
findstr /S /I "langdetect lingua" \*.py

# Linux / macOS
grep -rn "langdetect\\|lingua" --include="\*.py" .
```

Drop the unused one from `requirements.txt` to slim the install.

### 9.3 Four OCR backends

`easyocr`, `manga-ocr`, `pytesseract`, `paddleocr` (+ `paddlenlp`) are all
installed and used by `ocr\_processor.OCRFusion.best\_ocr()` for fusion /
fallback. **Don't remove any** unless you've confirmed via grep that
nothing calls that specific backend.

`pytesseract` also needs the **Tesseract binary** on `PATH` (not a pip
package). If you see `TesseractNotFoundError`:

* **Windows:** https://github.com/UB-Mannheim/tesseract/wiki → installer → add install dir to `PATH`
* **Linux:** `sudo apt install tesseract-ocr tesseract-ocr-jpn tesseract-ocr-kor`
* **macOS:** `brew install tesseract tesseract-lang`

### 9.4 `paddlenlp` import-order warning

```
\[WARNING] - Detected that datasets module was imported before paddlenlp.
```

Caused by `transformers` (or `datasets`) loading before `paddlenlp`. This
can break paddle OCR model downloads on some networks. Fix: in
`ocr\_processor.py`, put `import paddlenlp` **above** `import transformers`.

Cosmetic if your OCR works — safe to ignore in that case.

### 9.5 `cairosvg` on Windows

`cairosvg` needs **Cairo** native DLLs. If `import cairosvg` works today,
you're fine (Conda or a GTK install provides them). On a clean Windows
machine:

* Install the **GTK3 runtime**: https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases
* Or replace the SVG path in `magi\_server.base64\_to\_cv2()` with `svglib` + `reportlab`

### 9.6 `pkg\_resources` deprecation

```
UserWarning: pkg\_resources is deprecated as an API.
```

Comes from `jieba` (a `paddlenlp` dep). Silencing it:

```bash
pip install "setuptools<81"
```

### 9.7 Source bugs you should patch

These crash the corresponding code path when hit:

**`bulk\_processor.py`** — `\_create\_output\_zip()` calls `json.dumps(...)` but
`json` is never imported:

```python
# Add to top of file:
import json
```

**`magi\_server.py`** — calls `app.logger.error(...)` in several `except`
blocks. FastAPI's `app` has no `.logger` (that's a Flask habit):

```python
# Replace every:
app.logger.error(...)
# With the module-level logger already defined:
logger.error(...)
```

**`magi\_server.py`** — `run\_detection\_batch()` references globals
`PANEL\_MODEL` and `BUBBLE\_CHAR\_MODEL` that don't exist. Use the lazy
accessors:

```python
# Replace:
panel\_batch\_results  = PANEL\_MODEL(batch, ...)
bubble\_batch\_results = BUBBLE\_CHAR\_MODEL(batch, ...)

# With:
panel\_batch\_results  = get\_panel\_model()(batch, ...)
bubble\_batch\_results = get\_bubble\_char\_model()(batch, ...)
```

**`face\_processor.py`** — imports `FaceEmbeddingNet` which is never
defined (only `MangaEmbeddingNet` is). Currently unused by `magi\_server.py`,
so it doesn't break anything — but if you ever wire it up, it'll fail
at import time. Either delete the module or add the class definition.

\---

## 10\. Advanced: reproducible installs

Two files, two purposes:

|File|Role|
|-|-|
|`requirements.txt`|Human-curated, pinned **direct** deps. Readable. Explains *why* each package is here.|
|`requirements.lock.txt`|`pip freeze` snapshot — **everything**, including transitive deps. Byte-for-byte reproducibility.|

Generate the lock once your venv is working:

```bash
pip freeze > requirements.lock.txt
```

To reinstall on a fresh machine **exactly**:

```bash
python -m venv venv
venv\\Scripts\\activate                                  # Windows
# source venv/bin/activate                             # Linux/macOS
pip install -r requirements.lock.txt \\
  --extra-index-url https://download.pytorch.org/whl/cu128
python -m spacy download en\_core\_web\_sm
```

Same environment, guaranteed.

### 10.1 CI gate

Add this to your CI to prevent dependency drift:

```bash
python check\_deps.py --check-only
# exits 1 if any required dep is missing
```

### 10.2 Regenerate `requirements.txt` from current env

```bash
python check\_deps.py --write
```

Overwrites `requirements.txt` with pinned versions of whatever's currently
installed. Useful after upgrading torch, etc.

\---

## 11\. Linux / macOS notes

The default `requirements.txt` assumes Windows + CUDA 12.8. On Linux /
macOS you'll need these changes:

### 11.1 No NVIDIA GPU

Edit `requirements.txt`:

```diff
- torch==2.10.0+cu128
- torchvision==0.25.0+cu128
+ torch==2.10.0
+ torchvision==0.25.0
```

Then `pip install -r requirements.txt` (no extra index).

### 11.2 System packages

**Ubuntu / Debian**

```bash
sudo apt update
sudo apt install -y \\
    libcairo2            # cairosvg
    libgl1 libglib2.0-0  # opencv-python
    tesseract-ocr tesseract-ocr-jpn tesseract-ocr-kor  # pytesseract
    chromium-browser     # for Selenium
    xvfb                 # headless Chrome on servers without a display
```

**Fedora / RHEL**

```bash
sudo dnf install cairo libgl1 mesa-libGL tesseract chromium
```

**macOS**

```bash
brew install cairo tesseract tesseract-lang
brew install --cask google-chrome
```

### 11.3 Headless server (no display)

Wrap the backend and the browser in `xvfb-run`:

```bash
# Backend
xvfb-run -a python magi\_server.py

# Selenium captures on /capture\_image will need Chrome configured for headless (already done via --headless=new in image\_capture.py)
```

### 11.4 Chrome driver on headless server

`webdriver-manager` usually works. If it fails, install ChromeDriver
manually and set:

```bash
export CHROME\_DRIVER\_PATH=/usr/local/bin/chromedriver
```

\---

## 12\. One-page quick start

### Windows

```bat
:: ── 1. Setup (one time only) ─────────────────────────────
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt ^
  --extra-index-url https://download.pytorch.org/whl/cu128
python -m spacy download en\_core\_web\_sm

:: ── 2. Environment (each new terminal) ────────────────────
venv\\Scripts\\activate
set HF\_HOME=E:\\py\\hf\_cache
set TRANSFORMERS\_NO\_TF=1
set PYTORCH\_CUDA\_ALLOC\_CONF=expandable\_segments:True

:: ── 3. Verify ─────────────────────────────────────────────
python check\_deps.py --check-only
::   → must exit with code 0

:: ── 4. Terminal 1 — Node proxy ────────────────────────────
node server4.js

:: ── 5. Terminal 2 — FastAPI backend ───────────────────────
python magi\_server.py

:: ── 6. Browser ────────────────────────────────────────────
::   a) install Tampermonkey extension
::   b) import userscripts\\Manga\_OCR\_Pro\_tampermonkeylight.txt
::   c) allow your manga sites (@match)
::   d) open a manga chapter
::   e) click widget → OCR
```

### Linux / macOS

```bash
# ── 1. Setup ──────────────────────────────────────────────
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt \\
  --extra-index-url https://download.pytorch.org/whl/cu128
python -m spacy download en\_core\_web\_sm

# ── 2. Environment ────────────────────────────────────────
source venv/bin/activate
export HF\_HOME=$HOME/.cache/huggingface
export TRANSFORMERS\_NO\_TF=1
export PYTORCH\_CUDA\_ALLOC\_CONF=expandable\_segments:True

# ── 3. Verify ─────────────────────────────────────────────
python check\_deps.py --check-only

# ── 4. Terminal 1 — Node proxy ────────────────────────────
node server4.js

# ── 5. Terminal 2 — FastAPI backend ───────────────────────
python magi\_server.py

# ── 6. Browser: Tampermonkey + userscript (same as Windows)
```

### Startup order recap

```
Terminal 1:  node server4.js               → :3000
Terminal 2:  python magi\_server.py         → :5000
Browser:     Tampermonkey userscript
             → open manga site → click OCR
```

### The request flow

```
userscript ──► server4.js:3000/api/image-to-base64 ──► base64 image
     │
     └────► magi\_server.py:5000/detect  →  /ocr  →  /translate  →  /create\_overlay
```

Both endpoints must be reachable from the browser (both on `localhost`).

\---

**That's it.** If everything is running:

* `curl http://localhost:5000/system\_status` returns `{"backend\_healthy": true}`
* The widget appears on manga sites
* Clicking **OCR** overlays translations in \~10s (first run) then <1s
(cached)

If something isn't working, jump to **§8 Troubleshooting**.
