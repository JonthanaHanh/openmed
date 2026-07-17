"""Language-aware sentence segmentation utilities."""

from __future__ import annotations

import unicodedata
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..core.script_detect import is_grapheme_boundary, is_indic_text

# Python 3.12 emits SyntaxWarnings for old-style regex escapes in pysbd.
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

_SEGMENTER_CACHE: Dict[Tuple[str, bool], Any] = {}

_INDIC_TERMINATORS = frozenset({".", "!", "?", "।", "॥"})
_INDIC_SENTENCE_CONTINUATIONS = _INDIC_TERMINATORS | frozenset(
    {
        "'",
        '"',
        ")",
        "]",
        "}",
        "»",
        "’",
        "”",
        "›",
        "」",
        "』",
    }
)
_INDIC_HONORIFICS = frozenset(
    {
        "dr",
        "mr",
        "mrs",
        "ms",
        "prof",
        "डॉ",
        "डा",
        "श्री",
        "श्रीमती",
        "कु",
        "চি",
        "ডা",
        "ডাঃ",
        "ডঃ",
        "డా",
        "శ్రీ",
    }
)


@dataclass(frozen=True)
class SentenceSpan:
    """Represents a sentence and its character boundaries within the source."""

    text: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError("SentenceSpan requires 0 <= start <= end")


def _get_segmenter(
    *,
    language: str,
    clean: bool,
    segmenter: Optional[Any] = None,
) -> Any:
    """Return a cached pySBD segmenter instance."""
    if segmenter is not None:
        return segmenter

    cache_key = (language, clean)
    if cache_key in _SEGMENTER_CACHE:
        return _SEGMENTER_CACHE[cache_key]

    try:
        from pysbd import Segmenter  # type: ignore import
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError(
            "pySBD is required for sentence detection. "
            "Install it with `pip install pysbd` or add the `pysbd` dependency."
        ) from exc

    segmenter = Segmenter(
        language=language,
        clean=clean,
        char_span=True,
    )
    _SEGMENTER_CACHE[cache_key] = segmenter
    return segmenter


def _fallback_spans(text: str, sentences: Iterable[str]) -> List[SentenceSpan]:
    """Generate spans when pySBD does not provide char offsets."""
    spans: List[SentenceSpan] = []
    cursor = 0
    for sentence in sentences:
        if not sentence:
            continue

        start = text.find(sentence, cursor)
        if start == -1:
            stripped = sentence.strip()
            if stripped:
                start = text.find(stripped, cursor)
            if start == -1:
                start = cursor
        end = start + len(sentence)
        spans.append(SentenceSpan(sentence, start, end))
        cursor = end
    return spans


def _previous_word(text: str, terminator_index: int) -> str:
    cursor = terminator_index
    while cursor > 0:
        category = unicodedata.category(text[cursor - 1])
        if category[0] not in {"L", "M"}:
            break
        cursor -= 1
    return text[cursor:terminator_index].casefold()


def _next_nonspace(text: str, index: int) -> str:
    cursor = index + 1
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    return text[cursor] if cursor < len(text) else ""


def _is_guarded_terminator(text: str, index: int) -> bool:
    char = text[index]
    if (
        char == "."
        and index > 0
        and index + 1 < len(text)
        and text[index - 1].isdecimal()
        and text[index + 1].isdecimal()
    ):
        return True

    previous_word = _previous_word(text, index)
    if not previous_word:
        return False
    if char not in {".", "।"}:
        return False
    next_char = _next_nonspace(text, index)
    next_is_word = bool(next_char) and unicodedata.category(next_char)[0] in {
        "L",
        "M",
    }
    return next_is_word and (
        previous_word in _INDIC_HONORIFICS or (char == "." and len(previous_word) == 1)
    )


def _continues_indic_sentence(char: str) -> bool:
    return char.isspace() or char in _INDIC_SENTENCE_CONTINUATIONS


def segment_indic_text(text: str) -> List[SentenceSpan]:
    """Split Indic text on script-aware terminators with exact offsets.

    Danda and double-danda are treated as first-class sentence terminators.
    Common Indic and Latin honorifics, initials, and decimal points are guarded
    so embedded punctuation does not create a false boundary.
    """

    if not text:
        return []

    spans: List[SentenceSpan] = []
    start = 0
    boundary_ready = False

    for index, char in enumerate(text):
        if boundary_ready and not _continues_indic_sentence(char):
            if not (
                is_grapheme_boundary(start, text) and is_grapheme_boundary(index, text)
            ):
                raise ValueError("Indic sentence boundary splits a grapheme cluster")
            spans.append(SentenceSpan(text[start:index], start, index))
            start = index
            boundary_ready = False

        if char in _INDIC_TERMINATORS and not _is_guarded_terminator(text, index):
            boundary_ready = True

    if start < len(text):
        if not (
            is_grapheme_boundary(start, text) and is_grapheme_boundary(len(text), text)
        ):
            raise ValueError("Indic sentence boundary splits a grapheme cluster")
        spans.append(SentenceSpan(text[start:], start, len(text)))

    return spans


def segment_text(
    text: str,
    *,
    language: str = "en",
    clean: bool = False,
    segmenter: Optional[Any] = None,
) -> List[SentenceSpan]:
    """Split ``text`` into sentences and capture exact character spans.

    Text containing an Indic script uses the built-in danda-aware path. Purely
    Latin text and all other scripts retain the existing pySBD behavior.
    """
    if not text:
        return []

    if is_indic_text(text):
        return segment_indic_text(text)

    seg = _get_segmenter(language=language, clean=clean, segmenter=segmenter)
    sentences = seg.segment(text)

    spans: List[SentenceSpan] = []

    if sentences and hasattr(sentences[0], "start") and hasattr(sentences[0], "end"):
        for sentence in sentences:
            sent_text = getattr(sentence, "sent", None)
            if sent_text is None:
                sent_text = text[sentence.start : sentence.end]
            spans.append(
                SentenceSpan(
                    sent_text,
                    int(sentence.start),
                    int(sentence.end),
                )
            )
    else:
        spans = _fallback_spans(text, sentences)

    return spans


__all__ = ["SentenceSpan", "segment_indic_text", "segment_text"]
