"""Text processing utilities."""

import logging
import re
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Dict, List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Indic logical-order and NFC normalization
# ---------------------------------------------------------------------------

_DEVANAGARI_PREBASE_I = "\u093f"
_DEVANAGARI_NUKTA = "\u093c"
_DEVANAGARI_VIRAMA = "\u094d"
_INDIC_JOIN_CONTROLS = frozenset({"\u200c", "\u200d"})


@dataclass(frozen=True)
class IndicNormalization:
    """Logically ordered NFC text with a source alignment.

    ``char_origins[i]`` identifies the half-open source span that produced
    normalized character ``i``. The source units are code points by default,
    but callers such as the legacy-encoding converter may provide byte spans.
    """

    text: str
    char_origins: tuple[tuple[int, int], ...]

    def to_original_span(self, start: int, end: int) -> tuple[int, int]:
        """Map a normalized span to the supplied source coordinate system."""

        if not (0 <= start <= end <= len(self.text)):
            raise ValueError("span must satisfy 0 <= start <= end <= len(text)")
        if start == end:
            if start < len(self.char_origins):
                anchor = self.char_origins[start][0]
            elif self.char_origins:
                anchor = self.char_origins[-1][1]
            else:
                anchor = 0
            return anchor, anchor
        spans = self.char_origins[start:end]
        return min(span[0] for span in spans), max(span[1] for span in spans)


def normalize_indic_text(
    text: str,
    *,
    char_origins: tuple[tuple[int, int], ...] | None = None,
) -> IndicNormalization:
    """Normalize legacy Devanagari ordering and return stable NFC text.

    ISCII itself stores dependent matras in logical order, but visual-order
    legacy fonts often place vowel sign I before a consonant cluster. This
    function moves that pre-base glyph token after its logical cluster, puts a
    nukta before a following virama, expands canonically decomposed Devanagari
    letters, and produces NFC. Join controls following virama are retained.

    Args:
        text: Unicode text to normalize.
        char_origins: Optional source span for each input character.

    Returns:
        Idempotent normalized text and its source alignment.
    """

    if char_origins is None:
        char_origins = tuple((index, index + 1) for index in range(len(text)))
    if len(char_origins) != len(text):
        raise ValueError("char_origins must contain one span per input character")

    units: list[tuple[str, tuple[int, int]]] = []
    for char, origin in zip(text, char_origins):
        # Several Devanagari compatibility letters have canonical
        # decompositions that NFC intentionally retains. Expanding each unit
        # first makes nukta/virama ordering explicit and deterministic.
        for normalized_char in unicodedata.normalize("NFC", char):
            units.append((normalized_char, origin))

    reordered: list[tuple[str, tuple[int, int]]] = []
    index = 0
    while index < len(units):
        char = units[index][0]
        if (
            char == _DEVANAGARI_PREBASE_I
            and index + 1 < len(units)
            and _is_devanagari_consonant(units[index + 1][0])
        ):
            cluster_end = _devanagari_cluster_end(units, index + 1)
            reordered.extend(units[index + 1 : cluster_end])
            reordered.append(units[index])
            index = cluster_end
            continue
        reordered.append(units[index])
        index += 1

    # Unicode canonical ordering requires nukta (CCC 7) before virama (CCC 9).
    # Do the swap explicitly so malformed legacy sequences become valid input
    # even on runtimes whose normalization data predates a character.
    index = 1
    while index < len(reordered):
        if (
            reordered[index][0] == _DEVANAGARI_NUKTA
            and reordered[index - 1][0] == _DEVANAGARI_VIRAMA
        ):
            reordered[index - 1], reordered[index] = (
                reordered[index],
                reordered[index - 1],
            )
        index += 1

    normalized_units = _normalize_unit_groups(reordered)
    return IndicNormalization(
        text="".join(char for char, _ in normalized_units),
        char_origins=tuple(origin for _, origin in normalized_units),
    )


def _is_devanagari_consonant(char: str) -> bool:
    codepoint = ord(char)
    return 0x0915 <= codepoint <= 0x0939 or 0x0978 <= codepoint <= 0x097F


def _devanagari_cluster_end(
    units: list[tuple[str, tuple[int, int]]],
    start: int,
) -> int:
    index = start + 1
    if index < len(units) and units[index][0] == _DEVANAGARI_NUKTA:
        index += 1
    while index < len(units) and units[index][0] == _DEVANAGARI_VIRAMA:
        index += 1
        if index < len(units) and units[index][0] in _INDIC_JOIN_CONTROLS:
            index += 1
        if index >= len(units) or not _is_devanagari_consonant(units[index][0]):
            break
        index += 1
        if index < len(units) and units[index][0] == _DEVANAGARI_NUKTA:
            index += 1
    return index


def _normalize_unit_groups(
    units: list[tuple[str, tuple[int, int]]],
) -> list[tuple[str, tuple[int, int]]]:
    normalized: list[tuple[str, tuple[int, int]]] = []
    group: list[tuple[str, tuple[int, int]]] = []

    def flush() -> None:
        if not group:
            return
        source_text = "".join(char for char, _ in group)
        nfc_text = unicodedata.normalize("NFC", source_text)
        if nfc_text == source_text:
            normalized.extend(group)
        else:
            combined = (
                min(origin[0] for _, origin in group),
                max(origin[1] for _, origin in group),
            )
            normalized.extend((char, combined) for char in nfc_text)
        group.clear()

    for unit in units:
        char = unit[0]
        if (
            group
            and unicodedata.combining(char) == 0
            and char not in _INDIC_JOIN_CONTROLS
        ):
            flush()
        group.append(unit)
    flush()
    return normalized


class TextProcessor:
    """Handles text preprocessing and cleaning for medical text analysis."""

    def __init__(
        self,
        lowercase: bool = False,
        remove_punctuation: bool = False,
        remove_numbers: bool = False,
        normalize_whitespace: bool = True,
    ):
        """Initialize text processor.

        Args:
            lowercase: Whether to convert text to lowercase.
            remove_punctuation: Whether to remove punctuation.
            remove_numbers: Whether to remove numbers.
            normalize_whitespace: Whether to normalize whitespace.
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.normalize_whitespace = normalize_whitespace

        # Medical abbreviations that should be preserved
        self.medical_abbreviations = {
            "mg",
            "ml",
            "kg",
            "lb",
            "oz",
            "cm",
            "mm",
            "hr",
            "min",
            "bp",
            "hr",
            "rr",
            "temp",
            "o2",
            "co2",
            "hiv",
            "aids",
            "icu",
            "er",
            "or",
            "cbc",
            "ekg",
            "ecg",
            "mri",
            "ct",
            "x-ray",
            "ultrasound",
            "bmi",
            "copd",
            "chf",
            "mi",
            "stroke",
            "tia",
            "dvt",
            "pe",
            "uti",
            "copd",
        }

    def clean_text(self, text: str) -> str:
        """Clean and preprocess text.

        Args:
            text: Input text to clean.

        Returns:
            Cleaned text.
        """
        if not isinstance(text, str):
            text = str(text)

        original_text = text

        # Normalize whitespace
        if self.normalize_whitespace:
            text = re.sub(r"\s+", " ", text.strip())

        # Handle medical abbreviations before other processing
        protected_abbrevs = {}
        if not self.remove_punctuation:
            for i, abbrev in enumerate(self.medical_abbreviations):
                placeholder = f"__ABBREV_{i}__"
                text = re.sub(
                    rf"\b{re.escape(abbrev)}\b", placeholder, text, flags=re.IGNORECASE
                )
                protected_abbrevs[placeholder] = abbrev

        # Remove or clean numbers
        if self.remove_numbers:
            # Preserve medical measurements (e.g., "120/80", "98.6°F")
            text = re.sub(r"\b\d+(?:[./]\d+)*\b(?![°%])", " ", text)

        # Remove punctuation
        if self.remove_punctuation:
            # Keep hyphens in compound medical terms
            text = re.sub(r"[^\w\s\-]", " ", text)

        # Convert to lowercase
        if self.lowercase:
            text = text.lower()

        # Restore protected abbreviations
        for placeholder, abbrev in protected_abbrevs.items():
            text = text.replace(placeholder, abbrev)

        # Final whitespace normalization
        if self.normalize_whitespace:
            text = re.sub(r"\s+", " ", text.strip())

        logger.debug(
            "Text cleaning completed: input_chars=%d output_chars=%d changed=%s",
            len(original_text),
            len(text),
            original_text != text,
        )
        return text

    def segment_sentences(self, text: str) -> List[str]:
        """Segment text into sentences using medical text-aware rules.

        Args:
            text: Input text to segment.

        Returns:
            List of sentences.
        """
        # Medical abbreviations that shouldn't trigger sentence breaks
        abbrev_pattern = r"\b(?:" + "|".join(self.medical_abbreviations) + r")\."

        # Temporarily replace medical abbreviations
        text_modified = re.sub(
            abbrev_pattern,
            lambda m: m.group().replace(".", "___DOT___"),
            text,
            flags=re.IGNORECASE,
        )

        # Simple sentence segmentation
        sentences = re.split(r"[.!?]+\s+", text_modified)

        # Restore dots in abbreviations
        sentences = [s.replace("___DOT___", ".") for s in sentences if s.strip()]

        return sentences

    def extract_medical_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract basic medical entities using regex patterns.

        Args:
            text: Input text.

        Returns:
            Dictionary of entity types and their matches.
        """
        entities = {
            "medications": [],
            "dosages": [],
            "vital_signs": [],
            "lab_values": [],
            "symptoms": [],
        }

        # Dosage patterns
        dosage_patterns = [
            r"\b\d+\s*(?:mg|ml|g|kg|mcg|units?)\b",
            r"\b\d+\.\d+\s*(?:mg|ml|g|kg|mcg|units?)\b",
        ]

        for pattern in dosage_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities["dosages"].extend(matches)

        # Vital signs patterns
        vital_patterns = [
            r"\b(?:bp|blood pressure):?\s*\d+/\d+\b",
            r"\b(?:hr|heart rate):?\s*\d+\b",
            r"\b(?:temp|temperature):?\s*\d+\.?\d*\s*[°]?[fF]?\b",
            r"\b(?:rr|respiratory rate):?\s*\d+\b",
        ]

        for pattern in vital_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities["vital_signs"].extend(matches)

        # Clean up duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))

        return entities


def preprocess_text(
    text: str,
    lowercase: bool = False,
    remove_punctuation: bool = False,
    remove_numbers: bool = False,
    normalize_whitespace: bool = True,
) -> str:
    """Convenience function for text preprocessing.

    Args:
        text: Input text.
        lowercase: Whether to convert to lowercase.
        remove_punctuation: Whether to remove punctuation.
        remove_numbers: Whether to remove numbers.
        normalize_whitespace: Whether to normalize whitespace.

    Returns:
        Preprocessed text.
    """
    processor = TextProcessor(
        lowercase=lowercase,
        remove_punctuation=remove_punctuation,
        remove_numbers=remove_numbers,
        normalize_whitespace=normalize_whitespace,
    )
    return processor.clean_text(text)


def postprocess_text(text: str, capitalize_first: bool = True) -> str:
    """Postprocess text for better readability.

    Args:
        text: Input text.
        capitalize_first: Whether to capitalize the first letter.

    Returns:
        Postprocessed text.
    """
    if not text:
        return text

    text = text.strip()

    if capitalize_first and text:
        text = text[0].upper() + text[1:]

    return text


# ---------------------------------------------------------------------------
# Indic native-digit folding
# ---------------------------------------------------------------------------

# Base code point of the contiguous 0..9 decimal block for each supported
# Indic script. Only positional decimal digits are folded; non-positional
# number signs (e.g. the Tamil traditional signs U+0BF0-0BF2) are left as-is.
_INDIC_DIGIT_BASES: Dict[str, int] = {
    "devanagari": 0x0966,
    "bengali": 0x09E6,
    "gurmukhi": 0x0A66,
    "gujarati": 0x0AE6,
    "odia": 0x0B66,
    "tamil": 0x0BE6,
    "telugu": 0x0C66,
    "kannada": 0x0CE6,
    "malayalam": 0x0D66,
}

#: The Indic scripts whose decimal digits are folded to ASCII.
INDIC_DIGIT_SCRIPTS = tuple(_INDIC_DIGIT_BASES)

# Translation table from every native decimal digit code point to its ASCII
# equivalent. Built once; folding is a single str.translate call.
_INDIC_DIGIT_TRANSLATION = {
    base + digit: ord(str(digit))
    for base in _INDIC_DIGIT_BASES.values()
    for digit in range(10)
}


@dataclass(frozen=True)
class DigitFolding:
    """Native-digit text folded to ASCII, with a source offset mapping.

    Indic decimal digits are single code points, so folding is strictly
    length-preserving and the offset mapping is the identity: a span detected on
    the folded ``text`` indexes the same characters in ``original``, whose native
    digits are preserved for the output.
    """

    text: str
    original: str

    def to_original_span(self, start: int, end: int) -> "tuple[int, int]":
        """Map a folded ``[start, end)`` span to the original text offsets."""

        if not (0 <= start <= end <= len(self.text)):
            raise ValueError("span must satisfy 0 <= start <= end <= len(text)")
        return start, end


def fold_indic_digits(text: str) -> DigitFolding:
    """Fold Indic native decimal digits to ASCII, preserving offsets.

    Maps all nine native digit sets (Devanagari, Bengali, Gurmukhi, Gujarati,
    Odia, Tamil, Telugu, Kannada, Malayalam) to ASCII while leaving every other
    character -- ASCII digits, letters, and non-positional number signs --
    untouched. Folding is idempotent and length-preserving; the returned
    :class:`DigitFolding` keeps the original surface so native digits survive in
    the output.
    """

    return DigitFolding(text=text.translate(_INDIC_DIGIT_TRANSLATION), original=text)


def detect_with_digit_folding(
    text: str,
    matcher: Callable[[str], Iterable[Sequence[object]]],
) -> "list[tuple[object, ...]]":
    """Run ``matcher`` on digit-folded ``text`` and map spans back to the source.

    ``matcher`` takes the folded text and returns items whose first two elements
    are the ``(start, end)`` span; any trailing elements are preserved. Each
    result is ``(original_start, original_end, *trailing)``, so ASCII-only
    validators and regexes detect native-digit PHI while spans still index the
    original native-digit text.
    """

    folding = fold_indic_digits(text)
    results: "list[tuple[object, ...]]" = []
    for item in matcher(folding.text):
        start = int(item[0])  # type: ignore[call-overload]
        end = int(item[1])  # type: ignore[call-overload]
        original_start, original_end = folding.to_original_span(start, end)
        results.append((original_start, original_end, *tuple(item[2:])))
    return results
