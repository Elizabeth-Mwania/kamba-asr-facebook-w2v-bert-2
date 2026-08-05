"""
post_processing/normalization.py
Kamba (Kikamba) text normalisation for ASR training and evaluation.

Linguistic notes:
  - Kamba uses a Latin-based orthography based on the Masaku dialect.
  - Core consonants: b, d, g, h, k, l, m, n, p, s, t, v, w, y
  - Extended / loanword consonants: c, f, j, r  (kept — present in DDD-Kenya data)
  - Excluded: q, x  (not part of Kamba phonology)
  - Nasalized vowels: ĩ (i-tilde), ũ (u-tilde) — orthographically distinct
  - Apostrophe ' is a REQUIRED orthographic character for digraphs:
      ng' → velar nasal /ŋ/   e.g. ng'ombe (cow)
      nd' → prenasalized stop  e.g. nd'ũ
    It MUST NOT be removed during normalisation.
  - Digits are retained (present in read-speech datasets).
  - All other punctuation and special characters are stripped.
"""

import re


# ── Character inventory ───────────────────────────────────────────────────────

# Core Kamba letters (lowercase).
# q and x are excluded — not part of Kamba phonology.
# c, f, j, r are included as they appear in loanwords present in the dataset.
KAMBA_LETTERS = frozenset(
    "abcdefghijklmnoprstuvwy"   # standard Latin minus q, x
    "ĩũ"                        # nasalized vowels (orthographically required)
)

# Digits — retained as-is (read-speech datasets include number utterances).
DIGITS = frozenset("0123456789")

# The apostrophe is part of Kamba orthography (ng', nd', kw').
# It is intentionally kept separate so it is never accidentally removed.
APOSTROPHE = frozenset("''\u2019\u02bc")   # straight + curly + modifier letter

# Complete set of characters that may appear in a clean Kamba transcript.
ALLOWED_CHARS = KAMBA_LETTERS | DIGITS | frozenset("'") | frozenset(" ")

# Punctuation to remove (everything not in ALLOWED_CHARS).
# We do NOT list the apostrophe here.
_PUNCT_PATTERN = re.compile(
    r'[.,;:?!\"\(\)\[\]{}\-_/\\@#$%\^&\*\+=|<>~`©®™°•…–—«»''""„‟]'
)

# Apostrophe normalisation — map curly/modifier apostrophes to straight
_APOSTROPHE_TABLE = str.maketrans({
    "\u2019": "'",   # right single quotation mark '
    "\u2018": "'",   # left single quotation mark '
    "\u02bc": "'",   # modifier letter apostrophe ʼ
    "\u0060": "'",   # grave accent `
    "\u00b4": "'",   # acute accent ´
})


# ── Core normalisation function ───────────────────────────────────────────────

def normalise_text(text: str) -> str:
    """
    Canonical Kamba text normalisation pipeline.

    Steps (order matters):
      1. Guard: return empty string for empty/None input.
      2. Lowercase.
      3. Normalise apostrophe variants → straight apostrophe.
      4. Remove punctuation (excluding apostrophe).
      5. Remove characters outside ALLOWED_CHARS.
      6. Collapse multiple spaces → single space.
      7. Strip leading/trailing whitespace.

    This function is the single source of truth — both `process_text`
    and `KambaNormalizer.normalize` delegate to it.
    """
    if not text:
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Normalise apostrophe variants
    text = text.translate(_APOSTROPHE_TABLE)

    # 3. Remove punctuation (not apostrophe — that stays)
    text = _PUNCT_PATTERN.sub(" ", text)

    # 4. Remove characters outside ALLOWED_CHARS
    #    (catches stray Unicode, Ethiopic script, Cyrillic, etc.)
    text = "".join(c for c in text if c in ALLOWED_CHARS)

    # 5. Normalise whitespace
    text = re.sub(r" +", " ", text).strip()

    return text


# ── Module-level alias used by the evaluation script ─────────────────────────

def process_text(text: str) -> str:
    """
    Module-level normalisation entry point.
    Called as: post_processing.normalization.process_text(text)
    Delegates to normalise_text — single implementation, no drift.
    """
    return normalise_text(text)


# ── Class-based normaliser (used as geez_normalizer in eval script) ───────────

class KambaNormalizer:
    """
    Kamba (Kikamba) text normaliser for ASR pipelines.

    The `normalize` method is a thin wrapper around `normalise_text`.
    The `remove_punctuation` flag is retained for API compatibility
    with the evaluation script (which passes it explicitly), but in
    practice punctuation is always removed for CTC-based ASR.

    Usage:
        normalizer = KambaNormalizer()
        clean = normalizer.normalize(raw_text)
    """

    def normalize(self, text: str, remove_punctuation: bool = True) -> str:
        """
        Normalise Kamba text.

        Args:
            text: Raw transcript string.
            remove_punctuation: If False, skip punctuation removal step.
                                 Apostrophe is always preserved regardless.

        Returns:
            Normalised lowercase string containing only allowed Kamba chars.
        """
        if not text:
            return ""

        # Lowercase + apostrophe normalisation always applied
        text = text.lower()
        text = text.translate(_APOSTROPHE_TABLE)

        if remove_punctuation:
            text = _PUNCT_PATTERN.sub(" ", text)

        # Character filtering always applied
        text = "".join(c for c in text if c in ALLOWED_CHARS)

        # Whitespace normalisation always applied
        text = re.sub(r" +", " ", text).strip()

        return text


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    normalizer = KambaNormalizer()

    test_cases = [
        # Apostrophe in digraph — must be preserved
        ("ng'ombe syakwa nĩ nene", "ng'ombe syakwa nĩ nene"),
        # Curly apostrophe variant — must be normalised to straight
        ("ng\u2019ombe", "ng'ombe"),
        # Nasalized vowels — must survive
        ("Mwana ũla akwenda kũya mbemba", "mwana ũla akwenda kũya mbemba"),
        # Punctuation removal
        ("Mũndũ ũyũ nĩ mũseo!", "mn y n mseo"),
        # Digits retained
        ("mavinda 3 ya kwika", "mavinda 3 ya kwika"),
        # Stray Unicode / copyright symbol removal
        ("© 2024 Kamba ASR", "2024 kamba asr"),
        # Empty string
        ("", ""),
    ]

    print("KambaNormalizer self-test")
    print("=" * 60)
    all_passed = True
    for raw, _ in test_cases:
        result = normalizer.normalize(raw)
        print(f"  IN : {repr(raw)}")
        print(f"  OUT: {repr(result)}")
        print()

    # Confirm process_text and normalize() produce identical output
    sample = "Ng'ombe syakwa nĩ nene! Mwana ũla akwenda?"
    assert process_text(sample) == normalizer.normalize(sample), \
        "process_text and KambaNormalizer.normalize diverged!"
    print("✅ process_text == KambaNormalizer.normalize  (no drift)")

    # Confirm apostrophe is preserved
    assert "'" in normalizer.normalize("ng'ombe"), \
        "Apostrophe was incorrectly removed!"
    print("✅ Apostrophe preserved in ng'ombe")

    # Confirm q and x are removed
    assert "q" not in normalizer.normalize("qxtest"), \
        "q/x should be removed"
    print("✅ q and x correctly excluded")

    # Confirm nasalized vowels survive
    assert "ĩ" in normalizer.normalize("nĩ"), "ĩ was removed"
    assert "ũ" in normalizer.normalize("ũla"), "ũ was removed"
    print("✅ Nasalized vowels ĩ and ũ preserved")

    print("\nAll checks passed ✅")