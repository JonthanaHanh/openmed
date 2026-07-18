"""Unicode script detection helpers for mixed-script PII routing.

The helpers in this module are intentionally lightweight and stdlib-only. They
use explicit Unicode block ranges plus :mod:`unicodedata` character categories
to identify dominant scripts and preserve exact offsets while segmenting text
into script-oriented runs.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass

UNKNOWN_SCRIPT = "Unknown"

SUPPORTED_SCRIPTS = (
    "Latin",
    "Arabic",
    "Han",
    "Hiragana/Katakana",
    "Hangul",
    "Cyrillic",
    "Devanagari",
    "Tamil",
    "Telugu",
    "Greek",
    "Hebrew",
    "Thai",
)

SCRIPT_LANGUAGE_HINTS: dict[str, tuple[str, ...]] = {
    "Latin": ("en", "fr", "de", "it", "es", "nl", "pt", "tr"),
    "Arabic": ("ar",),
    "Han": ("ja",),
    "Hiragana/Katakana": ("ja",),
    "Hangul": ("ko",),
    "Cyrillic": ("en",),
    "Devanagari": ("hi",),
    "Tamil": ("hi",),
    "Telugu": ("te",),
    "Greek": ("en",),
    "Hebrew": ("en",),
    "Thai": ("en",),
    UNKNOWN_SCRIPT: ("en",),
}

ZERO_WIDTH_CHARS = frozenset(
    {
        "\u200b",  # zero width space
        "\u200c",  # zero width non-joiner
        "\u200d",  # zero width joiner
        "\u2060",  # word joiner
        "\ufeff",  # zero width no-break space
    }
)

_CONFUSABLE_FOLD: dict[str, str] = {
    "\u0391": "A",
    "\u0392": "B",
    "\u0395": "E",
    "\u0397": "H",
    "\u0399": "I",
    "\u039a": "K",
    "\u039c": "M",
    "\u039d": "N",
    "\u039f": "O",
    "\u03a1": "P",
    "\u03a4": "T",
    "\u03a7": "X",
    "\u03b1": "a",
    "\u03b5": "e",
    "\u03b7": "n",
    "\u03b9": "i",
    "\u03ba": "k",
    "\u03bc": "u",
    "\u03bf": "o",
    "\u03c1": "p",
    "\u03c4": "t",
    "\u03c5": "u",
    "\u03c7": "x",
    "\u0410": "A",
    "\u0412": "B",
    "\u0415": "E",
    "\u041a": "K",
    "\u041c": "M",
    "\u041d": "H",
    "\u041e": "O",
    "\u0420": "P",
    "\u0421": "C",
    "\u0422": "T",
    "\u0425": "X",
    "\u0430": "a",
    "\u0435": "e",
    "\u043e": "o",
    "\u0440": "p",
    "\u0441": "c",
    "\u0445": "x",
    "\u0456": "i",
}

INDIAN_NAME_LANGUAGES = frozenset({"hi", "ta"})
INDIAN_NAME_SCRIPTS = frozenset({"Devanagari", "Tamil"})

_DEVANAGARI_CONSONANTS = {
    "क": "k",
    "ख": "kh",
    "ग": "g",
    "घ": "gh",
    "ङ": "n",
    "च": "ch",
    "छ": "chh",
    "ज": "j",
    "झ": "jh",
    "ञ": "n",
    "ट": "t",
    "ठ": "th",
    "ड": "d",
    "ढ": "dh",
    "ण": "n",
    "त": "t",
    "थ": "th",
    "द": "d",
    "ध": "dh",
    "न": "n",
    "प": "p",
    "फ": "ph",
    "ब": "b",
    "भ": "bh",
    "म": "m",
    "य": "y",
    "र": "r",
    "ल": "l",
    "व": "v",
    "श": "sh",
    "ष": "sh",
    "स": "s",
    "ह": "h",
    "क़": "k",
    "ख़": "kh",
    "ग़": "g",
    "ज़": "j",
    "ड़": "d",
    "ढ़": "dh",
    "फ़": "f",
}
_DEVANAGARI_VOWELS = {
    "अ": "a",
    "आ": "aa",
    "इ": "i",
    "ई": "ii",
    "उ": "u",
    "ऊ": "uu",
    "ऋ": "r",
    "ए": "e",
    "ऐ": "ai",
    "ओ": "o",
    "औ": "au",
}
_DEVANAGARI_VOWEL_SIGNS = {
    "ा": "aa",
    "ि": "i",
    "ी": "ii",
    "ु": "u",
    "ू": "uu",
    "ृ": "r",
    "े": "e",
    "ै": "ai",
    "ो": "o",
    "ौ": "au",
}

_TAMIL_CONSONANTS = {
    "க": "k",
    "ங": "n",
    "ச": "s",
    "ஞ": "n",
    "ட": "t",
    "ண": "n",
    "த": "t",
    "ந": "n",
    "ப": "p",
    "ம": "m",
    "ய": "y",
    "ர": "r",
    "ல": "l",
    "வ": "v",
    "ழ": "l",
    "ள": "l",
    "ற": "r",
    "ன": "n",
    "ஜ": "j",
    "ஷ": "sh",
    "ஸ": "s",
    "ஹ": "h",
}
_TAMIL_VOWELS = {
    "அ": "a",
    "ஆ": "aa",
    "இ": "i",
    "ஈ": "ii",
    "உ": "u",
    "ஊ": "uu",
    "எ": "e",
    "ஏ": "e",
    "ஐ": "ai",
    "ஒ": "o",
    "ஓ": "o",
    "ஔ": "au",
}
_TAMIL_VOWEL_SIGNS = {
    "ா": "aa",
    "ி": "i",
    "ீ": "ii",
    "ு": "u",
    "ூ": "uu",
    "ெ": "e",
    "ே": "e",
    "ை": "ai",
    "ொ": "o",
    "ோ": "o",
    "ௌ": "au",
}

_DEVANAGARI_RENDER_CONSONANTS = {
    "kh": "ख",
    "gh": "घ",
    "ch": "च",
    "jh": "झ",
    "th": "थ",
    "dh": "ध",
    "ph": "फ",
    "bh": "भ",
    "sh": "श",
    "k": "क",
    "g": "ग",
    "c": "च",
    "j": "ज",
    "t": "त",
    "d": "द",
    "n": "न",
    "p": "प",
    "b": "ब",
    "m": "म",
    "y": "य",
    "r": "र",
    "l": "ल",
    "v": "व",
    "w": "व",
    "s": "स",
    "h": "ह",
    "f": "फ",
    "q": "क",
    "x": "क्स",
    "z": "ज",
}
_DEVANAGARI_RENDER_VOWELS = {
    "a": ("अ", ""),
    "i": ("इ", "ि"),
    "u": ("उ", "ु"),
    "e": ("ए", "े"),
    "o": ("ओ", "ो"),
    "ai": ("ऐ", "ै"),
    "au": ("औ", "ौ"),
}

_TAMIL_RENDER_CONSONANTS = {
    "kh": "க",
    "gh": "க",
    "ch": "ச",
    "jh": "ஜ",
    "th": "த",
    "dh": "த",
    "ph": "ப",
    "bh": "ப",
    "sh": "ஷ",
    "k": "க",
    "g": "க",
    "c": "ச",
    "j": "ஜ",
    "t": "த",
    "d": "த",
    "n": "ந",
    "p": "ப",
    "b": "ப",
    "m": "ம",
    "y": "ய",
    "r": "ர",
    "l": "ல",
    "v": "வ",
    "w": "வ",
    "s": "ஸ",
    "h": "ஹ",
    "f": "ஃப",
    "q": "க",
    "x": "க்ஸ",
    "z": "ஜ",
}
_TAMIL_RENDER_VOWELS = {
    "a": ("அ", ""),
    "i": ("இ", "ி"),
    "u": ("உ", "ு"),
    "e": ("எ", "ெ"),
    "o": ("ஒ", "ொ"),
    "ai": ("ஐ", "ை"),
    "au": ("ஔ", "ௌ"),
}

_PHONETIC_VOWEL_FOLDS = (
    ("ee", "i"),
    ("ii", "i"),
    ("oo", "u"),
    ("uu", "u"),
    ("aa", "a"),
)


@dataclass(frozen=True)
class DetectionNormalization:
    """Offset-preserving Unicode normalization for PII detection."""

    text: str
    original_length: int
    offset_starts: tuple[int, ...]
    offset_ends: tuple[int, ...]
    removed_zero_width: int = 0
    stripped_combining_marks: int = 0
    folded_confusables: int = 0
    folded_native_digits: int = 0
    scripts: tuple[str, ...] = ()
    mixed_script: bool = False

    @property
    def changed(self) -> bool:
        """Return whether the normalized text differs structurally."""
        return (
            self.removed_zero_width > 0
            or self.stripped_combining_marks > 0
            or self.folded_confusables > 0
            or self.folded_native_digits > 0
        )

    def remap_span(self, start: int, end: int) -> tuple[int, int]:
        """Map normalized-text offsets back to original-text offsets."""
        safe_start = max(0, min(int(start), len(self.text)))
        safe_end = max(safe_start, min(int(end), len(self.text)))
        if not self.offset_starts:
            return 0, 0
        if safe_start >= len(self.offset_starts):
            original_start = self.original_length
        else:
            original_start = self.offset_starts[safe_start]
        if safe_end <= 0:
            original_end = original_start
        elif safe_end - 1 >= len(self.offset_ends):
            original_end = self.original_length
        else:
            original_end = self.offset_ends[safe_end - 1]
        return original_start, max(original_start, original_end)

    def to_metadata(self) -> dict[str, object]:
        """Return PHI-free normalization metadata."""
        return {
            "changed": self.changed,
            "folded_confusables": self.folded_confusables,
            "folded_native_digits": self.folded_native_digits,
            "mixed_script": self.mixed_script,
            "removed_zero_width": self.removed_zero_width,
            "scripts": list(self.scripts),
            "stripped_combining_marks": self.stripped_combining_marks,
        }


_SCRIPT_RANGES: tuple[tuple[str, tuple[tuple[int, int], ...]], ...] = (
    (
        "Latin",
        (
            (0x0041, 0x005A),
            (0x0061, 0x007A),
            (0x00C0, 0x00FF),
            (0x0100, 0x017F),
            (0x0180, 0x024F),
            (0x1E00, 0x1EFF),
            (0x2C60, 0x2C7F),
            (0xA720, 0xA7FF),
            (0xAB30, 0xAB6F),
            (0xFF21, 0xFF3A),
            (0xFF41, 0xFF5A),
        ),
    ),
    (
        "Arabic",
        (
            (0x0600, 0x06FF),
            (0x0750, 0x077F),
            (0x08A0, 0x08FF),
            (0xFB50, 0xFDFF),
            (0xFE70, 0xFEFF),
        ),
    ),
    (
        "Han",
        (
            (0x3400, 0x4DBF),
            (0x4E00, 0x9FFF),
            (0xF900, 0xFAFF),
            (0x20000, 0x2A6DF),
            (0x2A700, 0x2B73F),
            (0x2B740, 0x2B81F),
            (0x2B820, 0x2CEAF),
            (0x2CEB0, 0x2EBEF),
            (0x30000, 0x3134F),
            (0x31350, 0x323AF),
        ),
    ),
    (
        "Hiragana/Katakana",
        (
            (0x3040, 0x309F),
            (0x30A0, 0x30FF),
            (0x31F0, 0x31FF),
            (0x1B000, 0x1B16F),
            (0xFF65, 0xFF9F),
        ),
    ),
    (
        "Hangul",
        (
            (0x1100, 0x11FF),
            (0x3130, 0x318F),
            (0xA960, 0xA97F),
            (0xAC00, 0xD7AF),
            (0xD7B0, 0xD7FF),
        ),
    ),
    (
        "Cyrillic",
        (
            (0x0400, 0x04FF),
            (0x0500, 0x052F),
            (0x1C80, 0x1C8F),
            (0x2DE0, 0x2DFF),
            (0xA640, 0xA69F),
        ),
    ),
    (
        "Devanagari",
        (
            (0x0900, 0x097F),
            (0xA8E0, 0xA8FF),
            (0x11B00, 0x11B5F),
        ),
    ),
    ("Tamil", ((0x0B80, 0x0BFF),)),
    ("Telugu", ((0x0C00, 0x0C7F),)),
    (
        "Greek",
        (
            (0x0370, 0x03FF),
            (0x1F00, 0x1FFF),
        ),
    ),
    (
        "Hebrew",
        (
            (0x0590, 0x05FF),
            (0xFB1D, 0xFB4F),
        ),
    ),
    ("Thai", ((0x0E00, 0x0E7F),)),
)


def indian_name_script(text: str, lang: str = "hi") -> str | None:
    """Return the rendering script for an Indian name surface, if in scope.

    Native Devanagari and Tamil surfaces are unambiguous. Latin surfaces opt in
    through a Hindi or Tamil language hint so unrelated Latin names retain their
    existing exact vault key behavior.
    """

    script = detect_script(text)
    if script in INDIAN_NAME_SCRIPTS:
        return script
    base_lang = str(lang or "").strip().replace("-", "_").split("_", 1)[0]
    if script == "Latin" and base_lang.casefold() in INDIAN_NAME_LANGUAGES:
        return script
    return None


def canonical_indian_name(text: str) -> str:
    """Fold a Devanagari, Tamil, or Romanized name to one phonetic key.

    This is a deterministic transliteration fold, not fuzzy entity resolution.
    It handles common long-vowel Roman variants and Indic consonant spellings
    while retaining the rest of each name, so similar but distinct names do not
    merge merely because they share a prefix.
    """

    script = detect_script(text)
    if script == "Devanagari":
        transliterated = _indic_to_latin(
            text,
            consonants=_DEVANAGARI_CONSONANTS,
            vowels=_DEVANAGARI_VOWELS,
            vowel_signs=_DEVANAGARI_VOWEL_SIGNS,
            virama="्",
            nasal_marks=frozenset({"ं", "ँ"}),
            ignored_marks=frozenset({"़"}),
        )
    elif script == "Tamil":
        transliterated = _indic_to_latin(
            text,
            consonants=_TAMIL_CONSONANTS,
            vowels=_TAMIL_VOWELS,
            vowel_signs=_TAMIL_VOWEL_SIGNS,
            virama="்",
            nasal_marks=frozenset(),
            ignored_marks=frozenset(),
        )
    else:
        transliterated = text
    return _fold_indian_romanization(transliterated)


def render_indian_name(canonical_name: str, script: str) -> str:
    """Render one canonical Indian surrogate identity in ``script``."""

    if script == "Devanagari":
        return _latin_to_indic(
            canonical_name,
            consonants=_DEVANAGARI_RENDER_CONSONANTS,
            vowels=_DEVANAGARI_RENDER_VOWELS,
            virama="्",
        )
    if script == "Tamil":
        return _latin_to_indic(
            canonical_name,
            consonants=_TAMIL_RENDER_CONSONANTS,
            vowels=_TAMIL_RENDER_VOWELS,
            virama="்",
        )
    return " ".join(part.capitalize() for part in canonical_name.split())


def _indic_to_latin(
    text: str,
    *,
    consonants: dict[str, str],
    vowels: dict[str, str],
    vowel_signs: dict[str, str],
    virama: str,
    nasal_marks: frozenset[str],
    ignored_marks: frozenset[str],
) -> str:
    output: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        consonant = consonants.get(char)
        if consonant is not None:
            following = text[index + 1] if index + 1 < len(text) else ""
            if following == virama:
                output.append(consonant)
                index += 2
                continue
            vowel_sign = vowel_signs.get(following)
            if vowel_sign is not None:
                output.append(consonant + vowel_sign)
                index += 2
                continue
            output.append(consonant + "a")
        elif char in vowels:
            output.append(vowels[char])
        elif char in nasal_marks:
            output.append("n")
        elif char in ignored_marks or char == virama:
            pass
        elif char.isascii() or char.isspace():
            output.append(char)
        index += 1

    words = "".join(output).split()
    return " ".join(word[:-1] if word.endswith("a") else word for word in words)


def _fold_indian_romanization(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text).casefold()
    ascii_letters = "".join(
        char
        for char in decomposed
        if not unicodedata.combining(char) and (char.isascii() or char.isspace())
    )
    folded = re.sub(r"[^a-z]+", " ", ascii_letters).strip()
    folded = folded.replace("sh", "s")
    for source, replacement in _PHONETIC_VOWEL_FOLDS:
        folded = folded.replace(source, replacement)
    return " ".join(folded.split())


def _latin_to_indic(
    text: str,
    *,
    consonants: dict[str, str],
    vowels: dict[str, tuple[str, str]],
    virama: str,
) -> str:
    consonant_tokens = sorted(consonants, key=len, reverse=True)
    vowel_tokens = sorted(vowels, key=len, reverse=True)
    output: list[str] = []
    index = 0
    previous_was_consonant = False

    while index < len(text):
        char = text[index]
        if not char.isalpha():
            if previous_was_consonant:
                output.append(virama)
            output.append(char)
            previous_was_consonant = False
            index += 1
            continue

        consonant = next(
            (token for token in consonant_tokens if text.startswith(token, index)),
            None,
        )
        if consonant is not None:
            if previous_was_consonant:
                output.append(virama)
            output.append(consonants[consonant])
            previous_was_consonant = True
            index += len(consonant)
            continue

        vowel = next(
            (token for token in vowel_tokens if text.startswith(token, index)),
            None,
        )
        if vowel is not None:
            independent, sign = vowels[vowel]
            output.append(sign if previous_was_consonant else independent)
            previous_was_consonant = False
            index += len(vowel)
            continue

        output.append(char)
        previous_was_consonant = False
        index += 1

    if previous_was_consonant:
        output.append(virama)
    return "".join(output)


def detect_script(text: str) -> str:
    """Return the dominant Unicode script in ``text``.

    Neutral characters such as whitespace, punctuation, symbols, and digits do
    not affect the decision. If no supported script-bearing code point is
    present, ``"Unknown"`` is returned.
    """

    counts: dict[str, int] = {}
    first_seen: dict[str, int] = {}

    for index, char in enumerate(text):
        script = _script_for_char(char)
        if script is None:
            continue
        counts[script] = counts.get(script, 0) + 1
        first_seen.setdefault(script, index)

    if not counts:
        return UNKNOWN_SCRIPT

    return max(counts, key=lambda script: (counts[script], -first_seen[script]))


def segment_by_script(text: str) -> Iterator[tuple[int, int, str]]:
    """Yield contiguous ``(start, end, script)`` runs covering ``text``.

    Neutral characters are assigned to the surrounding run: leading neutral
    characters attach to the first detected script, and later neutral characters
    attach to the preceding script. This keeps offsets exact while avoiding
    stand-alone whitespace or punctuation runs.
    """

    if not text:
        return

    run_start = 0
    current_script: str | None = None

    for index, char in enumerate(text):
        script = _script_for_char(char)
        if script is None:
            continue
        if current_script is None:
            current_script = script
            continue
        if script == current_script:
            continue

        yield run_start, index, current_script
        run_start = index
        current_script = script

    if current_script is None:
        yield 0, len(text), UNKNOWN_SCRIPT
        return

    yield run_start, len(text), current_script


def candidate_languages_for_script(script: str) -> tuple[str, ...]:
    """Return candidate language codes for a detected script."""

    return SCRIPT_LANGUAGE_HINTS.get(script, SCRIPT_LANGUAGE_HINTS[UNKNOWN_SCRIPT])


def normalize_for_pii_detection(
    text: str,
    *,
    width_convention: str = "cjk",
) -> DetectionNormalization:
    """Fold adversarial Unicode artifacts while preserving offset remapping.

    The defense strips zero-width controls and standalone combining marks, folds
    common Latin-lookalike Greek/Cyrillic/full-width characters, folds Indic
    decimal digits for ASCII validators, and records a script-consistency
    summary without storing source text. ``width_convention`` selects the
    CJK-safe width fold or strict per-character NFKC normalization.
    """

    # Keep the reusable width-normalization API in ``processing`` while
    # composing its explicit source map with this existing detection defense.
    # The local import avoids making the lightweight script helpers import the
    # broader processing package during module initialization.
    from ..processing.text import fold_indic_digits
    from ..processing.zh_normalize import normalize_width

    scripts = tuple(sorted(_script_counts(text)))
    width_normalization = normalize_width(text, convention=width_convention)
    digit_folding = fold_indic_digits(width_normalization.text)
    output: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    removed_zero_width = 0
    stripped_combining_marks = 0
    normalized_by_source: list[list[str]] = [[] for _ in text]
    for char, (original_start, _original_end) in zip(
        width_normalization.text,
        width_normalization.char_origins,
    ):
        normalized_by_source[original_start].append(char)
    changed_source_indices = {
        index
        for index, (char, normalized_chars) in enumerate(
            zip(text, normalized_by_source)
        )
        if "".join(normalized_chars) != char
    }
    folded_native_digit_sources = {
        width_normalization.char_origins[index][0]
        for index, (width_char, folded_char) in enumerate(
            zip(width_normalization.text, digit_folding.text)
        )
        if width_char != folded_char
    }

    for index, char in enumerate(digit_folding.text):
        original_start, original_end = width_normalization.char_origins[index]
        if char in ZERO_WIDTH_CHARS:
            removed_zero_width += 1
            continue
        if unicodedata.category(char) == "Mn":
            stripped_combining_marks += 1
            continue

        replacement = _fold_confusable_char(char)
        if replacement != char:
            changed_source_indices.add(original_start)
        for replacement_char in replacement:
            output.append(replacement_char)
            starts.append(original_start)
            ends.append(original_end)

    return DetectionNormalization(
        text="".join(output),
        original_length=len(text),
        offset_starts=tuple(starts),
        offset_ends=tuple(ends),
        removed_zero_width=removed_zero_width,
        stripped_combining_marks=stripped_combining_marks,
        folded_confusables=len(changed_source_indices),
        folded_native_digits=len(folded_native_digit_sources),
        scripts=scripts,
        mixed_script=len(scripts) > 1,
    )


def _script_for_char(char: str) -> str | None:
    category = unicodedata.category(char)
    if category[0] not in {"L", "M"}:
        return None

    codepoint = ord(char)
    for script, ranges in _SCRIPT_RANGES:
        if any(start <= codepoint <= end for start, end in ranges):
            return script
    return None


def _script_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for char in text:
        script = _script_for_char(char)
        if script is None:
            continue
        counts[script] = counts.get(script, 0) + 1
    return counts


def _fold_confusable_char(char: str) -> str:
    folded = _CONFUSABLE_FOLD.get(char)
    if folded is not None:
        return folded

    codepoint = ord(char)
    if 0xFF01 <= codepoint <= 0xFF5E:
        return chr(codepoint - 0xFEE0)

    return char


__all__ = [
    "DetectionNormalization",
    "INDIAN_NAME_LANGUAGES",
    "INDIAN_NAME_SCRIPTS",
    "SCRIPT_LANGUAGE_HINTS",
    "SUPPORTED_SCRIPTS",
    "UNKNOWN_SCRIPT",
    "ZERO_WIDTH_CHARS",
    "canonical_indian_name",
    "candidate_languages_for_script",
    "detect_script",
    "indian_name_script",
    "normalize_for_pii_detection",
    "render_indian_name",
    "segment_by_script",
]
