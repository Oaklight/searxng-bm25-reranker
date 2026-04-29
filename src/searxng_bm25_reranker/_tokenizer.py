"""CJK-aware tokenizer for BM25 reranking.

Handles mixed CJK + Latin text without external dependencies.
Latin text is tokenized by word boundaries; CJK character sequences
are split into unigrams + bigrams for better recall.
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[\w]+", re.UNICODE)

# CJK Unified Ideographs ranges
_CJK_RANGES = (
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0x2E80, 0x2EFF),  # CJK Radicals Supplement
    (0x3000, 0x303F),  # CJK Symbols and Punctuation
    (0x31C0, 0x31EF),  # CJK Strokes
    (0x3200, 0x32FF),  # Enclosed CJK Letters and Months
)


def _is_cjk(char: str) -> bool:
    """Check if a character is a CJK ideograph."""
    cp = ord(char)
    return any(start <= cp <= end for start, end in _CJK_RANGES)


def _has_cjk(text: str) -> bool:
    """Quick check if text contains any CJK characters."""
    return any(_is_cjk(c) for c in text)


def cjk_tokenize(text: str) -> list[str]:
    """Tokenize text with CJK awareness.

    For Latin/ASCII tokens: standard word-level tokenization.
    For CJK sequences: unigram + bigram decomposition.

    Example:
        "Redis缓存穿透" -> ["redis", "缓", "存", "穿", "透", "缓存", "存穿", "穿透"]
        "python async" -> ["python", "async"]

    Args:
        text: Input text to tokenize.

    Returns:
        List of lowercase tokens.
    """
    text = text.lower()
    tokens: list[str] = []

    for match in _WORD_RE.finditer(text):
        word = match.group()
        if _has_cjk(word):
            _tokenize_mixed(word, tokens)
        else:
            tokens.append(word)

    return tokens


def _tokenize_mixed(word: str, tokens: list[str]) -> None:
    """Split a mixed CJK/Latin word into segments and tokenize each."""
    cjk_buf: list[str] = []
    latin_buf: list[str] = []

    for char in word:
        if _is_cjk(char):
            if latin_buf:
                tokens.append("".join(latin_buf))
                latin_buf.clear()
            cjk_buf.append(char)
        else:
            if cjk_buf:
                _emit_cjk(cjk_buf, tokens)
                cjk_buf.clear()
            latin_buf.append(char)

    if latin_buf:
        tokens.append("".join(latin_buf))
    if cjk_buf:
        _emit_cjk(cjk_buf, tokens)


def _emit_cjk(chars: list[str], tokens: list[str]) -> None:
    """Emit unigrams and bigrams from a CJK character sequence."""
    # Unigrams
    for c in chars:
        tokens.append(c)
    # Bigrams
    for i in range(len(chars) - 1):
        tokens.append(chars[i] + chars[i + 1])
