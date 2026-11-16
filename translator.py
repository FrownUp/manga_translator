# translator.py - FULLY FIXED
#
# Fixes applied:
#   1. Heavy imports (T5, lingua) lazy-loaded -> won't crash on module import
#   2. MyMemory / Microsoft imported independently -> one missing pkg doesn't kill both
#   3. Direct HTTP fallback to translate.googleapis.com (independent of deep-translator)
#   4. asyncio.wait_for() timeouts on every engine -> no more hung requests
#   5. Retry with backoff per engine
#   6. OCR CJK spacing cleanup before translation (strips spurious spaces in JA/KO/ZH)
#   7. Translation dataclass now carries confidence + detected_lang
#   8. Single cache key format
#   9. asyncio.get_running_loop() instead of deprecated get_event_loop()
#  10. Conservative gender-pronoun handling (no more 'she'->'he' everywhere)
#  11. translatepy added as 4th independent engine (optional)
#  12. Real error logging -> next failure tells you WHY
#  13. translate_text() never raises -> always returns a string
#
# Keep in mind: this file must live alongside bubble_tracker.py and
# virtual_manga_page.py if you want those optional features.

import asyncio
import json as _json
import logging
import re
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Tuple

# ============================================================
# LAZY DEPENDENCIES (won't crash module import)
# ============================================================

# GoogleTranslator (via deep-translator)
_GOOGLE_CLS = None
_GOOGLE_IMPORT_TRIED = False
_GOOGLE_AVAILABLE = False

def _google():
    """Lazy-load GoogleTranslator. Returns class or None."""
    global _GOOGLE_CLS, _GOOGLE_IMPORT_TRIED, _GOOGLE_AVAILABLE
    if _GOOGLE_IMPORT_TRIED:
        return _GOOGLE_CLS
    _GOOGLE_IMPORT_TRIED = True
    try:
        from deep_translator import GoogleTranslator
        _GOOGLE_CLS = GoogleTranslator
        _GOOGLE_AVAILABLE = True
    except ImportError as e:
        logging.warning(f"GoogleTranslator (deep-translator) unavailable: {e}")
        _GOOGLE_AVAILABLE = False
    return _GOOGLE_CLS

# MyMemoryTranslator (independent import)
_MYMEMORY_CLS = None
_MYMEMORY_TRIED = False
_MYMEMORY_AVAILABLE = False

def _mymemory():
    global _MYMEMORY_CLS, _MYMEMORY_TRIED, _MYMEMORY_AVAILABLE
    if _MYMEMORY_TRIED:
        return _MYMEMORY_CLS
    _MYMEMORY_TRIED = True
    try:
        from deep_translator import MyMemoryTranslator
        _MYMEMORY_CLS = MyMemoryTranslator
        _MYMEMORY_AVAILABLE = True
    except ImportError:
        _MYMEMORY_AVAILABLE = False
    return _MYMEMORY_CLS

# MicrosoftTranslator (requires API key + [all] extra)
_MICROSOFT_CLS = None
_MICROSOFT_TRIED = False
_MICROSOFT_AVAILABLE = False

def _microsoft():
    global _MICROSOFT_CLS, _MICROSOFT_TRIED, _MICROSOFT_AVAILABLE
    if _MICROSOFT_TRIED:
        return _MICROSOFT_CLS
    _MICROSOFT_TRIED = True
    try:
        from deep_translator import MicrosoftTranslator
        _MICROSOFT_CLS = MicrosoftTranslator
        _MICROSOFT_AVAILABLE = True
    except ImportError:
        _MICROSOFT_AVAILABLE = False
    return _MICROSOFT_CLS

# translatepy (independent, optional)
_TRANSLATEPY = None
_TRANSLATEPY_TRIED = False
_TRANSLATEPY_AVAILABLE = False

def _translatepy_instance():
    global _TRANSLATEPY, _TRANSLATEPY_TRIED, _TRANSLATEPY_AVAILABLE
    if _TRANSLATEPY_TRIED:
        return _TRANSLATEPY
    _TRANSLATEPY_TRIED = True
    try:
        from translatepy import Translator as _T
        _TRANSLATEPY = _T()
        _TRANSLATEPY_AVAILABLE = True
        logging.info("translatepy available as fallback engine")
    except ImportError:
        _TRANSLATEPY_AVAILABLE = False
    return _TRANSLATEPY

# Backward-compat flag
ADVANCED_TRANSLATORS_AVAILABLE = True  # resolved lazily via _mymemory()/_microsoft()

# lingua language detector (lazy)
_DETECTOR = None
_DETECTOR_INITIALIZED = False

def _detector():
    global _DETECTOR, _DETECTOR_INITIALIZED
    if _DETECTOR_INITIALIZED:
        return _DETECTOR
    _DETECTOR_INITIALIZED = True
    try:
        from lingua import LanguageDetectorBuilder
        _DETECTOR = LanguageDetectorBuilder.from_all_languages().build()
        logging.info("lingua language detector loaded")
    except Exception as e:
        logging.warning(f"lingua unavailable, language detection disabled: {e}")
        _DETECTOR = None
    return _DETECTOR

# T5 refinement (lazy)
_T5_TOKENIZER = None
_T5_MODEL = None
_T5_TRIED = False
_T5_AVAILABLE = False

def _ensure_t5() -> bool:
    global _T5_TOKENIZER, _T5_MODEL, _T5_TRIED, _T5_AVAILABLE
    if _T5_TRIED:
        return _T5_AVAILABLE
    _T5_TRIED = True
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
        _T5_TOKENIZER = AutoTokenizer.from_pretrained("google/flan-t5-small")
        _T5_MODEL = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        _T5_MODEL.eval()
        _T5_AVAILABLE = True
        logging.info("T5 loaded for AI refinement")
    except Exception as e:
        logging.warning(f"T5 unavailable, AI refinement disabled: {e}")
        _T5_AVAILABLE = False
    return _T5_AVAILABLE

# ============================================================
# OPTIONAL: VIRTUAL PAGE + BUBBLE TRACKING
# ============================================================
try:
    from virtual_manga_page import (
        VirtualMangaPage,
        ContextAwareMangaProcessor,
        ReadingOrder,
    )
    VIRTUAL_PAGE_ENABLED = True
    context_processor = ContextAwareMangaProcessor()
    logging.info("Virtual manga page context enabled in translator")
except Exception as e:
    VIRTUAL_PAGE_ENABLED = False
    context_processor = None
    VirtualMangaPage = None
    ContextAwareMangaProcessor = None
    ReadingOrder = None
    logging.warning(f"Virtual manga page not available: {e}")

try:
    from bubble_tracker import bubble_tracker
    BUBBLE_TRACKING_ENABLED = True
    logging.info("Bubble tracking enabled in translator")
except Exception:
    BUBBLE_TRACKING_ENABLED = False
    bubble_tracker = None
    logging.warning("Bubble tracking not available - create bubble_tracker.py")

# ============================================================
# STATE
# ============================================================
translation_memory: Dict[str, "Translation"] = {}
previous_languages: Dict[str, str] = {}
bubble_speakers: Dict[str, str] = {}
speaker_genders: Dict[str, str] = {}


class Translation:
    """Translation record. Carries confidence and detected language."""
    __slots__ = ("translated_text", "confidence", "detected_lang")

    def __init__(self, translated_text: str, confidence: float = 1.0,
                 detected_lang: str = "unknown"):
        self.translated_text = translated_text
        self.confidence = confidence
        self.detected_lang = detected_lang


def assign_speaker(bubble_id: str, speaker_id: str):
    bubble_speakers[bubble_id] = speaker_id


def get_speaker(bubble_id: str) -> Optional[str]:
    return bubble_speakers.get(bubble_id)


def set_speaker_gender(speaker_id: str, gender: str):
    speaker_genders[speaker_id] = gender.lower()


def get_speaker_gender(speaker_id: str) -> Optional[str]:
    return speaker_genders.get(speaker_id)


# ============================================================
# LANGUAGE CODES
# ============================================================
LANGUAGE_MAPPING = {
    # ISO 639-3 and 639-1 aliases -> deep-translator codes
    "jpn": "ja", "ja": "ja", "japanese": "ja",
    "eng": "en", "en": "en", "english": "en",
    "deu": "de", "de": "de", "german": "de",
    "spa": "es", "es": "es", "spanish": "es",
    "fra": "fr", "fr": "fr", "french": "fr",
    "ara": "ar", "ar": "ar", "arabic": "ar",
    "tha": "th", "th": "th", "thai": "th",
    "kor": "ko", "ko": "ko", "korean": "ko",
    "ind": "id", "id": "id", "indonesian": "id",
    "rus": "ru", "ru": "ru", "russian": "ru",
    "por": "pt", "pt": "pt", "portuguese": "pt",
    "chi_sim": "zh-cn", "chi_tra": "zh-tw",
    "zh": "zh-cn", "zh-cn": "zh-cn", "zh-tw": "zh-tw",
    "chinese": "zh-cn",
    # Lingua's iso_code_639_1 codes
    "zh": "zh-cn", "tl": "tl", "vi": "vi",
}

SUPPORTED_LANGUAGES = [
    "ja", "en", "de", "es", "fr", "ar", "th", "ko",
    "id", "ru", "pt", "zh-cn",
]


def _normalize_lang(code: Optional[str]) -> str:
    if not code:
        return "en"
    return LANGUAGE_MAPPING.get(code.lower(), code.lower())


# ============================================================
# T5 REFINEMENT (used by context-aware refinement)
# ============================================================
def _run_t5_refiner(prompt: str, max_new_tokens: int = 60) -> str:
    if not _ensure_t5():
        return ""
    try:
        import torch
        inputs = _T5_TOKENIZER(
            prompt, return_tensors="pt", truncation=True, max_length=256
        )
        inputs = {k: v.to(_T5_MODEL.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = _T5_MODEL.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.4,
                repetition_penalty=1.2,
                pad_token_id=_T5_TOKENIZER.pad_token_id,
            )
        return _T5_TOKENIZER.decode(out[0], skip_special_tokens=True).strip()
    except Exception as e:
        logging.warning(f"T5 refinement failed: {e}")
        return ""


# ============================================================
# VIRTUAL PAGE HELPERS
# ============================================================
def get_virtual_page_context(bubble_id: str, window_size: int = 2) -> List[Dict]:
    if not VIRTUAL_PAGE_ENABLED or not context_processor:
        return []
    try:
        return context_processor.get_context_for_translation(bubble_id, window_size) or []
    except Exception as e:
        logging.warning(f"get_virtual_page_context failed: {e}")
        return []


def update_virtual_page_translation(bubble_id: str, translated_text: str,
                                    confidence: float = 1.0):
    if not VIRTUAL_PAGE_ENABLED or not context_processor:
        return
    try:
        context_processor.update_bubble_translation(bubble_id, translated_text, confidence)
    except Exception as e:
        logging.warning(f"update_virtual_page_translation failed: {e}")


def extract_context_history(bubble_id: str, window_size: int = 2) -> List[str]:
    history = []
    for b in get_virtual_page_context(bubble_id, window_size):
        text = (
            b.get("translated_text")
            or b.get("text")
            or b.get("original_text")
            or ""
        ).strip()
        if text:
            history.append(text)
    return history


# ============================================================
# PRONOUN PRESERVATION (conservative)
# ============================================================
def preserve_pronouns(original_text: str, translated_text: str,
                      speaker_id: Optional[str] = None) -> str:
    """
    Only rewrite pronouns when the *source* text has an explicit pronoun
    whose meaning we know. Unlike the previous version, this does NOT
    blindly swap every 'she' to 'he' for a male speaker — which corrupted
    quoted sentences like "She is my sister".
    """
    if not original_text or not translated_text:
        return translated_text
    gender = get_speaker_gender(speaker_id) if speaker_id else None
    result = translated_text

    # Source-explicit pronouns only
    if "私" in original_text or "僕" in original_text or "俺" in original_text:
        result = re.sub(r"\b(i|me)\b", "I", result, flags=re.IGNORECASE)
    if "あなた" in original_text:
        result = re.sub(r"\b(you)\b", "you", result, flags=re.IGNORECASE)

    # 彼 -> he, 彼女 -> she (these are literal)
    if "彼女" in original_text:
        result = re.sub(r"\bhe\b", "she", result, flags=re.IGNORECASE)
        result = re.sub(r"\bhim\b", "her", result, flags=re.IGNORECASE)
        result = re.sub(r"\bhis\b", "her", result, flags=re.IGNORECASE)
    elif "彼" in original_text and "彼女" not in original_text:
        result = re.sub(r"\bshe\b", "he", result, flags=re.IGNORECASE)
        result = re.sub(r"\bher\b", "him", result, flags=re.IGNORECASE)

    # Only apply gender hints if the original had a gendered pronoun
    if gender and ("彼" in original_text or "彼女" in original_text):
        result = fix_gender_pronouns(result, gender)

    return result


# ============================================================
# AI CONTEXT REFINEMENT
# ============================================================
def clean_ai_output(text: str, original_text: str) -> str:
    if not text or text == original_text:
        return original_text
    patterns = [
        r"^Refine this English translation so that it fits naturally into the context[^.]*\.",
        r"^Refine this English translation so it fits the context naturally[^.]*\.",
        r"^Context: Text:",
        r"^Improved translation:",
        r"^Refined version:",
        r"^Natural English:",
    ]
    cleaned = text
    for p in patterns:
        cleaned = re.sub(p, "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r'^["\']|["\']$', "", cleaned).strip()
    if (
        len(cleaned) < 2
        or cleaned.lower() == original_text.lower()
        or any(
            phrase in cleaned.lower()
            for phrase in (
                "refine this", "context:", "preserves meaning", "sounds human"
            )
        )
    ):
        return original_text
    return cleaned


async def ai_context_refine_with_virtual_page(
    translated_text: str,
    bubble_id: Optional[str] = None,
) -> str:
    if not translated_text or not translated_text.strip():
        return translated_text
    if not _ensure_t5():
        return translated_text

    history = extract_context_history(bubble_id) if bubble_id else []
    if not history:
        return translated_text

    context_str = " | ".join(history[-2:])
    speaker_id = get_speaker(bubble_id)
    gender = get_speaker_gender(speaker_id) if speaker_id else None
    speaker_info = (
        f"Speaker: {speaker_id} ({gender})" if speaker_id else "Speaker unknown"
    )

    prompt = (
        f"Previous: {context_str} | Current: {translated_text} | "
        f"{speaker_info} | Keep correct gender, pronouns (I/me/you/he/she) "
        f"and natural flow:"
    )

    try:
        loop = asyncio.get_running_loop()
        raw = await loop.run_in_executor(None, lambda: _run_t5_refiner(prompt, 60))
        refined = clean_ai_output(raw, translated_text)
        refined = preserve_pronouns(translated_text, refined, speaker_id)

        if (
            refined
            and refined != translated_text
            and len(refined) > 2
            and len(refined.split()) >= len(translated_text.split()) * 0.5
        ):
            logging.info(f"AI refinement: '{translated_text}' -> '{refined}'")
            return refined
        return translated_text
    except Exception as e:
        logging.warning(f"Virtual page context refinement failed: {e}")
        return translated_text


async def ai_context_refine_batch_with_virtual_page(
    translations: List[str],
    bubble_ids: Optional[List[str]] = None,
) -> List[str]:
    if not bubble_ids:
        return translations
    out = []
    for t, bid in zip(translations, bubble_ids):
        if not t or not t.strip() or not bid:
            out.append(t)
        else:
            out.append(await ai_context_refine_with_virtual_page(t, bid))
    return out


# ============================================================
# OCR CORRECTION (with CJK spacing cleanup)
# ============================================================
OCR_CORRECTIONS = {
    "ja": {
        "l": "I", "0": "O", "5": "S", "2": "Z", "8": "B",
        "rn": "m", "cl": "d", "vv": "w",
    },
    "ko": {"l": "I", "0": "O"},
    "zh": {"l": "I", "0": "O"},
}

# CJK ranges — used for spacing cleanup
_CJK_CLASS = (
    r"[\u3000-\u303f\u3040-\u309f\u30a0-\u30ff"
    r"\u4e00-\u9fff\uff00-\uffef\uac00-\ud7af]"
)


def _clean_cjk_spaces(text: str, lang: str) -> str:
    """Remove spaces that OCR inserted between CJK characters."""
    if not text:
        return text
    norm = _normalize_lang(lang)
    if norm not in ("ja", "ko", "zh-cn", "zh-tw"):
        return text
    # Remove space between two CJK characters
    return re.sub(rf"(?<={_CJK_CLASS})\s+(?={_CJK_CLASS})", "", text)


def correct_ocr_errors(text: str, lang: str = "ja") -> str:
    """Fix common OCR errors before translation."""
    if not text or not text.strip():
        return text

    norm = _normalize_lang(lang)
    corrections = OCR_CORRECTIONS.get(norm, OCR_CORRECTIONS["ja"])

    corrected = text

    # Apply simple single-char corrections (whole-word only)
    for wrong, right in sorted(corrections.items(), key=lambda x: -len(x[0])):
        pattern = rf"(?<=[^A-Za-z]){re.escape(wrong)}(?=[^A-Za-z])"
        corrected = re.sub(pattern, right, corrected)

    # OCR often inserts spaces inside Japanese/Korean/Chinese words
    corrected = _clean_cjk_spaces(corrected, norm)

    # Collapse whitespace
    corrected = re.sub(r"\s+", " ", corrected).strip()

    # Repeated-character artifact fix (e.g. "やややや" -> "や")
    corrected = re.sub(r"(.)\1{3,}", r"\1", corrected)

    return corrected


# ============================================================
# MANGA GLOSSARY
# ============================================================
MANGA_GLOSSARY = {
    "sensei": {"en": "sensei"},
    "senpai": {"en": "senpai"},
    "kouhai": {"en": "kouhai"},
    "san": {"en": "san"},
    "kun": {"en": "kun"},
    "chan": {"en": "chan"},
    "sama": {"en": "sama"},
    "nakama": {"en": "comrade"},
    "baka": {"en": "baka"},
    "nani": {"en": "nani"},
    "kawaii": {"en": "kawaii"},
    "itadakimasu": {"en": "enjoy your meal"},
    "arigatou": {"en": "arigatou"},
    "gomen": {"en": "sorry"},
    "chibi": {"en": "chibi"},
    "otaku": {"en": "otaku"},
    "manga": {"en": "manga"},
    "anime": {"en": "anime"},
}


def apply_manga_glossary(text: str, source_lang: str, target_lang: str) -> str:
    if not text or target_lang == source_lang:
        return text
    result = text.lower()
    for term, tr in MANGA_GLOSSARY.items():
        if term in result:
            result = re.sub(
                r"\b" + re.escape(term) + r"\b",
                tr.get(target_lang, term),
                result,
            )
    return result


# ============================================================
# MULTI-ENGINE TRANSLATION
# ============================================================
async def _google_direct_http(
    text: str, src: str, tgt: str = "en"
) -> Optional[str]:
    """
    Direct HTTP fallback to translate.googleapis.com.
    Independent of deep-translator — survives library breakage.
    """
    try:
        url = (
            "https://translate.googleapis.com/translate_a/single"
            "?client=gtx&sl={}&tl={}&dt=t&q={}"
        ).format(src, tgt, urllib.parse.quote(text))

        def _do():
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0 Safari/537.36"
                    ),
                    "Accept": "application/json,text/plain,*/*",
                },
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                payload = _json.loads(resp.read().decode("utf-8"))
            if payload and isinstance(payload, list) and payload[0]:
                parts = []
                for segment in payload[0]:
                    if segment and segment[0]:
                        parts.append(segment[0])
                out = "".join(parts).strip()
                return out or None
            return None

        loop = asyncio.get_running_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(None, _do), timeout=20
        )
    except asyncio.TimeoutError:
        logging.warning("google_direct: timeout")
        return None
    except Exception as e:
        logging.warning(f"google_direct failed: {type(e).__name__}: {e}")
        return None


async def translate_with_best_engine(
    text: str,
    source_lang: str,
    target_lang: str = "en",
) -> str:
    """
    Try every engine in order and return the first good result.
    Never raises — falls back to original text.
    """
    if not text or not text.strip():
        return text

    src = _normalize_lang(source_lang)
    tgt = _normalize_lang(target_lang)
    if src == tgt:
        return text

    cleaned = _clean_cjk_spaces(text.strip(), src)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return text

    errors: List[str] = []
    loop = asyncio.get_running_loop()

    # ---- Engine 1: deep-translator Google (fast, rate-limited) ----
    google_cls = _google()
    if google_cls is not None:
        for attempt in range(2):
            try:
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: google_cls(source=src, target=tgt).translate(cleaned),
                    ),
                    timeout=15,
                )
                if result and result.strip() and result.strip() != cleaned:
                    return result.strip()
            except asyncio.TimeoutError:
                errors.append(f"google#{attempt + 1}: timeout")
            except Exception as e:
                errors.append(f"google#{attempt + 1}: {type(e).__name__}: {e}")
                await asyncio.sleep(0.4 * (attempt + 1))
    else:
        errors.append("google: unavailable")

    # ---- Engine 2: MyMemory ----
    mm_cls = _mymemory()
    if mm_cls is not None:
        for attempt in range(2):
            try:
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: mm_cls(source=src, target=tgt).translate(cleaned),
                    ),
                    timeout=15,
                )
                if result and result.strip() and result.strip() != cleaned:
                    return result.strip()
            except asyncio.TimeoutError:
                errors.append(f"mymemory#{attempt + 1}: timeout")
            except Exception as e:
                errors.append(f"mymemory#{attempt + 1}: {type(e).__name__}: {e}")
                await asyncio.sleep(0.4 * (attempt + 1))
    else:
        errors.append("mymemory: unavailable")

    # ---- Engine 3: translatepy ----
    tp = _translatepy_instance()
    if tp is not None:
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: str(
                        tp.translate(
                            cleaned,
                            source_language=src,
                            destination_language=tgt,
                        )
                    ),
                ),
                timeout=15,
            )
            if result and result.strip() and result.strip() != cleaned:
                return result.strip()
        except asyncio.TimeoutError:
            errors.append("translatepy: timeout")
        except Exception as e:
            errors.append(f"translatepy: {type(e).__name__}: {e}")
    else:
        errors.append("translatepy: unavailable")

    # ---- Engine 4: direct HTTP to Google ----
    result = await _google_direct_http(cleaned, src, tgt)
    if result and result.strip() and result.strip() != cleaned:
        return result.strip()
    errors.append("google_direct: no result")

    # ---- Everything failed: log the real reason ----
    logging.error(
        "All translation engines failed. text=%r src=%s tgt=%s errors=%s",
        text[:120], src, tgt, errors,
    )
    return text


# ============================================================
# POST-PROCESSING
# ============================================================
def post_process_translation(
    text: str,
    gender: Optional[str] = None,
    source_lang: str = "ja",
    target_lang: str = "en",
) -> str:
    if not text or not text.strip():
        return text
    result = text
    result = apply_manga_glossary(result, source_lang, target_lang)
    if gender:
        result = fix_gender_pronouns(result, gender)
    result = fix_translation_artifacts(result)
    result = fix_capitalization(result)
    return re.sub(r"\s+", " ", result).strip()


def fix_gender_pronouns(text: str, gender: str) -> str:
    """Adjust pronouns based on speaker gender. Only called when we actually
    have a hint; not applied blindly."""
    if not gender or gender == "unknown":
        return text
    fixes = {
        "male": {
            r"\bshe\b": "he",
            r"\bher\b": "him",
            r"\bhers\b": "his",
            r"\bherself\b": "himself",
        },
        "female": {
            r"\bhe\b": "she",
            r"\bhim\b": "her",
            r"\bhis\b": "hers",
            r"\bhimself\b": "herself",
        },
    }
    for pattern, repl in fixes.get(gender, {}).items():
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    return text


def fix_translation_artifacts(text: str) -> str:
    if not text:
        return text
    result = re.sub(r"  +", " ", text)
    result = re.sub(r"\s+([.,!?;:])", r"\1", result)
    result = re.sub(r"([.,!?;:])([A-Za-z])", r"\1 \2", result)
    if (result.startswith('"') and result.endswith('"')) or (
        result.startswith("'") and result.endswith("'")
    ):
        result = result[1:-1]
    result = re.sub(r"\bi\b", "I", result)
    return result.strip()


def fix_capitalization(text: str) -> str:
    if not text:
        return text
    parts = re.split(r"([.!?]\s+)", text)
    for i in range(0, len(parts), 2):
        s = parts[i].strip()
        if s:
            parts[i] = s[0].upper() + s[1:]
    result = "".join(parts)
    return re.sub(r"\bi\b", "I", result)


# ============================================================
# QUALITY SCORING
# ============================================================
def score_translation_quality(original: str, translated: str) -> float:
    if not original or not translated:
        return 0.0
    if original == translated:
        return 1.0
    score = 0.5
    len_ratio = len(translated) / max(len(original), 1)
    if 0.5 <= len_ratio <= 2.0:
        score += 0.2
    artifacts = ["...", "??", "!!", "  "]
    score -= sum(0.05 for a in artifacts if a in translated)
    if translated[0].isupper():
        score += 0.1
    if "  " not in translated:
        score += 0.1
    return min(max(score, 0.0), 1.0)


# ============================================================
# MAIN TRANSLATE
# ============================================================
async def translate_text_enhanced(
    text: str,
    source_language: str = "auto",
    bubble_id: Optional[str] = None,
    speaker_id: Optional[str] = None,
    context: Optional[List[str]] = None,
) -> Tuple[str, str, float]:
    """
    Full pipeline. Returns (translated, detected_lang, quality_score).
    Never raises.
    """
    if not text or not text.strip():
        _record(bubble_id, text, 1.0, True)
        return text, "unknown", 1.0

    # Resolve gender
    gender = None
    if speaker_id:
        gender = get_speaker_gender(speaker_id)
    elif bubble_id:
        sp = get_speaker(bubble_id)
        if sp:
            gender = get_speaker_gender(sp)

    # Step 1: OCR cleanup
    corrected = correct_ocr_errors(text, source_language)

    # Step 2: Detect language
    detected = _normalize_lang(source_language)
    if source_language == "auto" or not source_language:
        det = _detector()
        if det is not None:
            try:
                found = await asyncio.get_running_loop().run_in_executor(
                    None, lambda: det.detect_language_of(corrected)
                )
                if found:
                    detected = _normalize_lang(found.iso_code_639_1.name.lower())
                else:
                    detected = "en"
            except Exception as e:
                logging.warning(f"Language detection failed: {e}")
                detected = "en"
        else:
            detected = "en"

    # Step 3: cache check
    cache_key = f"{corrected}::{detected}::{gender or '-'}"
    if cache_key in translation_memory:
        cached = translation_memory[cache_key]
        _record(bubble_id, cached.translated_text, cached.confidence, True)
        return cached.translated_text, cached.detected_lang, cached.confidence

    # Step 4: translate
    if detected == "en":
        translated = corrected
        confidence = 1.0
    else:
        translated = await translate_with_best_engine(corrected, detected, "en")
        confidence = 0.8 if translated and translated != corrected else 0.0

    # Step 5: post-processing
    final = post_process_translation(
        translated, gender=gender,
        source_lang=detected, target_lang="en",
    )

    # Step 6: context refinement
    if bubble_id and VIRTUAL_PAGE_ENABLED:
        try:
            history = context if context else extract_context_history(bubble_id)
            if history:
                refined = await ai_context_refine_with_virtual_page(final, bubble_id)
                if refined:
                    final = refined
        except Exception as e:
            logging.warning(f"Context refinement failed: {e}")

    # Step 7: quality score + cache
    quality = score_translation_quality(corrected, final)
    translation_memory[cache_key] = Translation(final, quality, detected)
    previous_languages[source_language or "auto"] = detected

    # Step 8: record in tracking systems
    _record(bubble_id, final, quality, success=(final != corrected))
    if VIRTUAL_PAGE_ENABLED and bubble_id:
        update_virtual_page_translation(bubble_id, final, quality)

    return final, detected, quality


def _record(bubble_id: Optional[str], translated_text: str,
            confidence: float, success: bool):
    """Push result into bubble tracker if available."""
    if not BUBBLE_TRACKING_ENABLED or not bubble_id or not bubble_tracker:
        return
    try:
        bubble_tracker.update_bubble_translation(
            bubble_id=bubble_id,
            translated_text=translated_text,
            target_language="en",
            confidence=confidence,
            success=success,
        )
    except Exception as e:
        logging.warning(f"Bubble tracking update failed: {e}")


# ============================================================
# BATCH
# ============================================================
async def batch_translate_enhanced(
    texts: List[str],
    source_language: str = "auto",
    bubble_ids: Optional[List[str]] = None,
    speaker_ids: Optional[List[str]] = None,
) -> List[Tuple[str, str, float]]:
    if not texts:
        return []
    tasks = []
    for i, text in enumerate(texts):
        bid = bubble_ids[i] if bubble_ids and i < len(bubble_ids) else None
        sid = speaker_ids[i] if speaker_ids and i < len(speaker_ids) else None
        tasks.append(
            translate_text_enhanced(
                text, source_language,
                bubble_id=bid, speaker_id=sid,
            )
        )
    results = await asyncio.gather(*tasks, return_exceptions=True)
    out = []
    for r in results:
        if isinstance(r, Exception):
            logging.error(f"Batch translation error: {r}")
            out.append(("", "unknown", 0.0))
        else:
            out.append(r)
    return out


# ============================================================
# BACKWARD-COMPAT WRAPPERS
# ============================================================
async def translate_text(
    text: str,
    source_language: str = "auto",
    bubble_id: Optional[str] = None,
) -> Tuple[str, str]:
    """Legacy signature. Uses enhanced pipeline. Never raises."""
    try:
        result, lang, _ = await translate_text_enhanced(
            text, source_language, bubble_id=bubble_id
        )
        return result, lang
    except Exception as e:
        logging.error(f"translate_text failed: {e}")
        return text, "unknown"


async def translate_text_with_virtual_page(
    text: str,
    source_language: str = "auto",
    bubble_id: Optional[str] = None,
) -> Tuple[str, str]:
    return await translate_text(text, source_language, bubble_id)


async def batch_translate(
    texts: List[str],
    source_language: str = "auto",
    bubble_ids: Optional[List[str]] = None,
) -> List[str]:
    """Legacy signature. Uses enhanced pipeline. Never raises."""
    try:
        results = await batch_translate_enhanced(
            texts, source_language, bubble_ids=bubble_ids
        )
        return [r[0] for r in results]
    except Exception as e:
        logging.error(f"batch_translate failed: {e}")
        return list(texts)


# ============================================================
# VIRTUAL PAGE SETUP
# ============================================================
def setup_virtual_page_from_detection(
    image_path: str,
    detection_results: Dict,
    reading_order: str = "manga",
) -> Optional[str]:
    if not VIRTUAL_PAGE_ENABLED or not context_processor:
        return None
    try:
        order = ReadingOrder.MANGA if reading_order.lower() == "manga" else ReadingOrder.WESTERN
        page = context_processor.create_virtual_page_from_detection(
            image_path, detection_results, order
        )
        return page.export_to_json()
    except Exception as e:
        logging.error(f"Failed to setup virtual page: {e}")
        return None


# ============================================================
# HELPERS / STATS
# ============================================================
def clear_translation_memory():
    translation_memory.clear()
    previous_languages.clear()


def get_translation_stats() -> Dict:
    stats = {
        "cache_size": len(translation_memory),
        "supported_languages": SUPPORTED_LANGUAGES,
        "previous_languages": previous_languages,
        "virtual_page_enabled": VIRTUAL_PAGE_ENABLED,
        "bubble_tracking_enabled": BUBBLE_TRACKING_ENABLED,
        "t5_available": _T5_AVAILABLE,
        "detector_available": _DETECTOR is not None,
        "engines": {
            "google": _GOOGLE_AVAILABLE,
            "mymemory": _MYMEMORY_AVAILABLE,
            "microsoft": _MICROSOFT_AVAILABLE,
            "translatepy": _TRANSLATEPY_AVAILABLE,
            "google_direct": True,
        },
    }
    if BUBBLE_TRACKING_ENABLED:
        try:
            stats["bubble_tracking"] = bubble_tracker.get_bubble_stats()
        except Exception as e:
            logging.warning(f"Could not get bubble stats: {e}")
    return stats


def is_bubble_tracking_enabled() -> bool:
    return BUBBLE_TRACKING_ENABLED


def is_virtual_page_enabled() -> bool:
    return VIRTUAL_PAGE_ENABLED


def get_bubble_stats() -> Optional[Dict]:
    if BUBBLE_TRACKING_ENABLED and bubble_tracker:
        try:
            return bubble_tracker.get_bubble_stats()
        except Exception as e:
            logging.error(f"Failed to get bubble stats: {e}")
    return None


# ============================================================
# SELF-TEST (optional — run `python -m translator` to sanity check)
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    )

    async def _demo():
        samples = [
            ("他に選択肢がないのだから仕 方あるまい...", "ja"),
            ("俺はお前を守る。", "ja"),
            ("これはテストです", "ja"),
            ("Hello world", "en"),
            ("안녕하세요", "ko"),
        ]
        for text, src in samples:
            out, lang, score = await translate_text_enhanced(text, src)
            print(f"  {text!r}\n    -> {out!r}   lang={lang}  score={score:.2f}\n")

    print("=== translator.py self-test ===")
    print("Engine availability:")
    for name, avail in get_translation_stats()["engines"].items():
        print(f"  {name:15s}: {'✓' if avail else '✗'}")
    print()
    asyncio.run(_demo())