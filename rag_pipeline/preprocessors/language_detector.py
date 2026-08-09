"""
preprocessors/language_detector.py

Lightweight language detector that works fully offline — no external API needed.
Uses character n-gram profiles for English and Spanish (the two languages
observed in the OCR output). Falls back to "unknown" gracefully.

We intentionally avoid heavy libraries (langdetect spins up a JVM-like
process; fasttext requires a model download). This tiny heuristic is
accurate enough for EN/ES discrimination on paragraph-length text.
"""

import re
from collections import Counter


# ── Spanish-language signal words (high-frequency, low-ambiguity) ─────────────
_ES_SIGNALS = frozenset([
    "el", "la", "los", "las", "de", "del", "en", "es", "un", "una",
    "que", "con", "para", "por", "se", "su", "sus", "al", "son",
    "una", "como", "este", "esta", "pero", "más", "si", "ya", "también",
    "cuando", "porque", "tiene", "hay", "sido", "ser", "del", "entre",
    # domain-specific Spanish words from the OCR docs
    "definición", "botón", "conjunto", "trabajo", "posición", "función",
    "nombre", "tarea", "operativa", "restricciones", "informe", "sistema",
    "elemento", "datos", "valores", "cadena", "borrado", "arranque",
    "entrada", "salida", "acción", "operador",
])

# ── English-language signal words ─────────────────────────────────────────────
_EN_SIGNALS = frozenset([
    "the", "and", "for", "are", "with", "this", "that", "from", "have",
    "not", "but", "they", "been", "will", "can", "which", "when", "there",
    "their", "about", "would", "into", "more", "also", "only", "other",
    # domain words
    "operator", "action", "entry", "report", "restriction", "system",
    "service", "track", "parameter", "prompt", "model", "chunk",
])

_RE_WORD = re.compile(r'\b[a-záéíóúüñ]{2,}\b', re.IGNORECASE)


def detect_language(text: str) -> str:
    """
    Return "es", "en", or "unknown".

    Algorithm:
      - Tokenise into lowercase words (ignoring numbers & punctuation).
      - Count hits against ES and EN signal word sets.
      - The set with the higher hit-rate wins.
      - If the margin is too slim (< 20% relative difference), return "unknown".
    """
    words = [w.lower() for w in _RE_WORD.findall(text)]
    if len(words) < 5:
        return "unknown"

    total = len(words)
    es_hits = sum(1 for w in words if w in _ES_SIGNALS)
    en_hits = sum(1 for w in words if w in _EN_SIGNALS)

    es_rate = es_hits / total
    en_rate = en_hits / total

    if es_rate == 0 and en_rate == 0:
        return "unknown"

    # Require at least 20 % relative advantage to call a language
    if es_rate > en_rate:
        relative_margin = (es_rate - en_rate) / max(es_rate, 1e-9)
        return "es" if relative_margin >= 0.20 else "unknown"
    elif en_rate > es_rate:
        relative_margin = (en_rate - es_rate) / max(en_rate, 1e-9)
        return "en" if relative_margin >= 0.20 else "unknown"
    else:
        return "unknown"
