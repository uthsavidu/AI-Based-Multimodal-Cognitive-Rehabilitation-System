"""Text-side analysis for the Voice Cognitive Game.

This module contains only deterministic, explainable logic (word lists,
tokenization, set/multiset comparison). No ML model is used here because
none is needed -- word comparison and category-membership checking are a
better fit for simple deterministic methods than for a model (see project
guidance: don't add ML just to say the project uses AI).

Timing metrics (response time, speech duration, pause count/duration,
speech rate) are measured client-side using the Web Speech API plus a
lightweight Web Audio volume-based VAD, and are passed into these
functions/routes as already-measured values -- they are not invented here.
"""

from __future__ import annotations

import unicodedata
from difflib import SequenceMatcher

import voice_content

FILLER_WORDS = {"um", "umm", "uh", "uhh", "uhm", "er", "erm", "ah", "like", "hmm", "you know"}


def get_category_word_list(category: str, language: str | None = "en") -> set[str]:
    """Language-aware controlled vocabulary lookup for Level 3 categories.

    English, Tamil, Kannada and Hindi vocabularies live in voice_content.py
    (one place for all multilingual Voice Game content); this just delegates
    to it, keeping the category-matching algorithm below language-neutral.
    """
    return voice_content.category_word_list(language, (category or "").strip())


def _tokenize(text: str) -> list[str]:
    """Split spoken/recognized text into comparable word tokens.

    Unicode-category-based rather than a regex character-range whitelist:
    keeps letters (category L*) and combining marks (category M*) together
    so vowel signs stay attached to their base consonant -- this matters
    for Tamil/Kannada/Devanagari, where a regex like [^a-z0-9'] (or even
    a bare `\\w`, which excludes combining marks) would strip or split
    matras and silently turn words into isolated consonants. Numbers and a
    plain apostrophe are also kept; everything else (whitespace,
    punctuation) is a token boundary.
    """
    if not text:
        return []
    text = text.lower()
    tokens: list[str] = []
    current: list[str] = []
    for ch in text:
        category = unicodedata.category(ch)
        if category[0] in ("L", "M", "N") or ch == "'":
            current.append(ch)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens


def count_filler_words(text: str, language: str | None = "en") -> int:
    tokens = _tokenize(text)
    fillers = FILLER_WORDS | voice_content.EXTRA_FILLER_WORDS.get(
        voice_content.normalize_language(language), set()
    )
    return sum(1 for token in tokens if token in fillers)


def _levenshtein_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def score_single_word_recall(expected_word: str, recognized_text: str, language: str | None = "en") -> dict:
    """Level 1: single-word recall. Allows minor STT misrecognition tolerance."""
    expected = (expected_word or "").strip().lower()
    tokens = _tokenize(recognized_text)
    recognized_word = tokens[0] if tokens else ""

    exact_match = bool(recognized_word) and recognized_word == expected
    fuzzy_match = bool(recognized_word) and _levenshtein_ratio(recognized_word, expected) >= 0.8
    correct = exact_match or fuzzy_match

    return {
        "recognized_text": recognized_text or "",
        "correct": correct,
        "filler_count": count_filler_words(recognized_text, language),
    }


def score_sentence_recall(expected_sentence: str, recognized_text: str, language: str | None = "en") -> dict:
    """Level 2: sentence recall. Multiset word comparison + order similarity.

    "Word-order difference" is reported as a heuristic similarity ratio
    (difflib SequenceMatcher over the token sequences), not a linguistically
    validated syntax comparison -- documented here so it is not
    over-claimed elsewhere.
    """
    expected_tokens = _tokenize(expected_sentence)
    recognized_tokens = _tokenize(recognized_text)

    from collections import Counter
    expected_counts = Counter(expected_tokens)
    recognized_counts = Counter(recognized_tokens)

    correct_words = 0
    missing_words = 0
    for word, count in expected_counts.items():
        matched = min(count, recognized_counts.get(word, 0))
        correct_words += matched
        missing_words += count - matched

    extra_words = 0
    repeated_words = 0
    for word, count in recognized_counts.items():
        expected_count = expected_counts.get(word, 0)
        if count > expected_count:
            overflow = count - expected_count
            if expected_count == 0:
                extra_words += overflow
            else:
                repeated_words += overflow

    total_expected = max(1, len(expected_tokens))
    recall_accuracy = round((correct_words / total_expected) * 100.0, 2)
    order_similarity = round(_levenshtein_ratio(" ".join(expected_tokens), " ".join(recognized_tokens)) * 100.0, 2)

    return {
        "recognized_text": recognized_text or "",
        "correct_words": correct_words,
        "missing_words": missing_words,
        "extra_words": extra_words,
        "repeated_words": repeated_words,
        "recall_accuracy": recall_accuracy,
        "word_order_similarity_percent": order_similarity,
        "filler_count": count_filler_words(recognized_text, language),
    }


def score_category_round(category: str, recognized_text: str, language: str | None = "en") -> dict:
    """Level 3: one 15-second category word-generation round.

    Validates each recognized word against the controlled vocabulary for
    the category (in the selected language) rather than accepting every
    recognized word as valid.
    """
    vocabulary = get_category_word_list(category, language)
    tokens = _tokenize(recognized_text)
    fillers = FILLER_WORDS | voice_content.EXTRA_FILLER_WORDS.get(
        voice_content.normalize_language(language), set()
    )

    seen: set[str] = set()
    valid_words: list[str] = []
    unique_valid_words: list[str] = []
    repeated_words = 0
    invalid_words: list[str] = []

    for token in tokens:
        if token in fillers:
            continue
        if token in vocabulary:
            valid_words.append(token)
            if token in seen:
                repeated_words += 1
            else:
                seen.add(token)
                unique_valid_words.append(token)
        else:
            invalid_words.append(token)

    return {
        "recognized_text": recognized_text or "",
        "valid_word_count": len(valid_words),
        "unique_valid_word_count": len(unique_valid_words),
        "repeated_words": repeated_words,
        "invalid_word_count": len(invalid_words),
        "unique_valid_words": unique_valid_words,
        "filler_count": count_filler_words(recognized_text, language),
    }


def compute_round_trend(round_scores: list[int]) -> str:
    """Determine whether Level 3 word-generation performance trends up/down/flat/variable."""
    if len(round_scores) < 2:
        return "Not enough rounds to determine a trend"

    diffs = [round_scores[i + 1] - round_scores[i] for i in range(len(round_scores) - 1)]
    if all(diff > 0 for diff in diffs):
        return "Increasing"
    if all(diff < 0 for diff in diffs):
        return "Decreasing"
    if all(diff == 0 for diff in diffs):
        return "Stable"
    return "Variable"
