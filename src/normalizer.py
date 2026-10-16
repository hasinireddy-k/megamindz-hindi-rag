"""
Hindi and English Text Normalization Module.

Handles Devanagari-specific Unicode normalization (NFC), Nukta standardization,
Zero-Width Joiner/Non-Joiner cleaning, Danda normalization, and whitespace standardization.
"""

import re
import unicodedata
from typing import Literal

# Devanagari character ranges
DEVANAGARI_RANGE = re.compile(r'[\u0900-\u097F]')

# Devanagari composite Nukta mapping (decomposed -> canonical precomposed)
NUKTA_MAP = {
    '\u0915\u093C': '\u0958',  # ? + ? -> ??
    '\u0916\u093C': '\u0959',  # ? + ? -> ??
    '\u0917\u093C': '\u095A',  # ? + ? -> ??
    '\u091C\u093C': '\u095B',  # ? + ? -> ??
    '\u0921\u093C': '\u095C',  # ? + ? -> ??
    '\u0922\u093C': '\u095D',  # ? + ? -> ??
    '\u092B\u093C': '\u095E',  # ? + ? -> ??
    '\u092F\u093C': '\u095F',  # ? + ? -> ??
}

# Zero-width characters
ZERO_WIDTH_CHARS = re.compile(r'[\u200B\u200E\u200F\uFEFF]')
STRAY_ZWNJ = re.compile(r'(?<![\u0900-\u097F])\u200C|\u200C(?![\u0900-\u097F])')
STRAY_ZWJ = re.compile(r'(?<![\u0900-\u097F])\u200D|\u200D(?![\u0900-\u097F])')


def detect_language(text: str) -> Literal['hi', 'en', 'mixed']:
    """
    Detect whether the input string is predominantly Hindi (Devanagari), English, or Mixed.
    """
    if not text:
        return 'en'
    devanagari_count = len(DEVANAGARI_RANGE.findall(text))
    latin_count = len(re.findall(r'[a-zA-Z]', text))

    if devanagari_count > 0 and latin_count > 0:
        return 'mixed'
    if devanagari_count > 0:
        return 'hi'
    return 'en'


def normalize_text(text: str, preserve_newlines: bool = True) -> str:
    """
    Performs comprehensive Unicode, Hindi Devanagari, and whitespace normalization.

    Steps:
    1. Unicode Canonical Decomposition & Composition (NFC).
    2. Removal of non-printing zero-width characters (BOM, zero-width space).
    3. Normalization of stray Zero-Width Non-Joiners (ZWNJ) and Joiners (ZWJ).
    4. Standardization of Devanagari Nukta character pairs.
    5. Normalization of ASCII pipe '|' to Devanagari Purna Viram '?' when acting as sentence terminator.
    6. Normalization of varied quotes, dashes, and non-breaking spaces.
    7. Whitespace cleaning while optionally preserving structural paragraph breaks.
    """
    if not text:
        return ""

    # 1. Unicode Canonical Composition
    text = unicodedata.normalize('NFC', text)

    # 2. Strip invisible zero-width spaces & BOM
    text = ZERO_WIDTH_CHARS.sub('', text)
    text = STRAY_ZWNJ.sub('', text)
    text = STRAY_ZWJ.sub('', text)

    # 3. Canonicalize Nukta composites
    for decomposed, composed in NUKTA_MAP.items():
        text = text.replace(decomposed, composed)

    # 4. Standardize sentence terminators for Devanagari
    # Replace pipe '|' with Devanagari Danda '।' (\u0964) if preceded by Devanagari characters or whitespace
    text = re.sub(r'([\u0900-\u097F\s])\|', r'\g<1>।', text)

    # 5. Standardize quotes and non-breaking spaces
    text = text.replace('\u00A0', ' ')
    text = text.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
    text = text.replace('—', '-').replace('–', '-')

    # 6. Whitespace and newline management
    if preserve_newlines:
        # Preserve double newlines (paragraphs), collapse single spaces
        lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in text.splitlines()]
        # Remove consecutive blank lines (> 2)
        text = re.sub(r'\n{3,}', '\n\n', '\n'.join(lines)).strip()
    else:
        text = re.sub(r'\s+', ' ', text).strip()

    return text
