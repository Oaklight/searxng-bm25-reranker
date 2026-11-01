"""Tokenizer tests, with emphasis on the CJK ranges that were missing."""

from __future__ import annotations

from searxng_bm25_reranker._tokenizer import cjk_tokenize


class TestLatin:
    def test_splits_on_word_boundaries(self) -> None:
        assert cjk_tokenize("python async") == ["python", "async"]

    def test_lowercases(self) -> None:
        assert cjk_tokenize("Python ASYNC") == ["python", "async"]

    def test_punctuation_is_a_boundary(self) -> None:
        assert cjk_tokenize("foo-bar.baz") == ["foo", "bar", "baz"]

    def test_empty_and_punctuation_only(self) -> None:
        assert cjk_tokenize("") == []
        assert cjk_tokenize("   !!! ") == []


class TestChinese:
    def test_unigrams_and_bigrams(self) -> None:
        assert cjk_tokenize("缓存穿透") == [
            "缓",
            "存",
            "穿",
            "透",
            "缓存",
            "存穿",
            "穿透",
        ]

    def test_mixed_latin_and_cjk(self) -> None:
        assert cjk_tokenize("Redis缓存穿透") == [
            "redis",
            "缓",
            "存",
            "穿",
            "透",
            "缓存",
            "存穿",
            "穿透",
        ]


class TestJapanese:
    """Regression: hiragana and katakana were absent from _CJK_RANGES.

    Before the fix "とても簡単" tokenized to ['とても', '簡', '単', '簡単'],
    so a kana-only run never produced the unigrams/bigrams BM25 needs to match.
    """

    def test_hiragana_run_is_split(self) -> None:
        tokens = cjk_tokenize("とても簡単")
        assert "と" in tokens
        assert "て" in tokens
        assert "とても" not in tokens

    def test_kana_and_kanji_share_bigrams(self) -> None:
        assert "簡単" in cjk_tokenize("とても簡単")

    def test_katakana_is_split(self) -> None:
        tokens = cjk_tokenize("スクリプト")
        assert "スク" in tokens
        assert "スクリプト" not in tokens

    def test_kana_no_longer_glues_to_latin(self) -> None:
        """The old failure mode: 'なjavascript' as one unusable token."""
        tokens = cjk_tokenize("簡単なJavaScript")
        assert "javascript" in tokens
        assert "なjavascript" not in tokens
        assert "簡単" in tokens


class TestOtherScripts:
    def test_korean_stays_word_level(self) -> None:
        """Hangul is intentionally not a CJK range: Korean uses word spaces."""
        assert cjk_tokenize("한국어 검색") == ["한국어", "검색"]

    def test_fullwidth_latin_stays_a_word(self) -> None:
        tokens = cjk_tokenize("ＡＢＣ全角")
        assert "ａｂｃ" in tokens
        assert "全角" in tokens
