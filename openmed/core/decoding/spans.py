"""Span post-processing helpers shared across privacy-filter backends.

When token classifiers emit slightly-too-greedy spans (e.g. "alice@hospital.org and"
absorbs the trailing "and"), these helpers tighten the boundaries before the
span reaches downstream redaction logic. Pure-Python; no array-framework
dependencies.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import Any, Final

_INDIC_SCRIPT_RANGES: Final[tuple[tuple[int, int], ...]] = (
    (0x0900, 0x097F),  # Devanagari
    (0x0980, 0x09FF),  # Bengali
    (0x0A00, 0x0A7F),  # Gurmukhi
    (0x0A80, 0x0AFF),  # Gujarati
    (0x0B00, 0x0B7F),  # Odia
    (0x0B80, 0x0BFF),  # Tamil
    (0x0C00, 0x0C7F),  # Telugu
    (0x0C80, 0x0CFF),  # Kannada
    (0x0D00, 0x0D7F),  # Malayalam
)

_INDIC_VIRAMAS: Final[frozenset[int]] = frozenset(
    {
        0x094D,  # Devanagari sign virama
        0x09CD,  # Bengali sign virama
        0x0A4D,  # Gurmukhi sign virama
        0x0ACD,  # Gujarati sign virama
        0x0B4D,  # Odia sign virama
        0x0BCD,  # Tamil sign virama
        0x0C4D,  # Telugu sign virama
        0x0CCD,  # Kannada sign virama
        0x0D4D,  # Malayalam sign virama
    }
)

_JOIN_CONTROLS: Final[frozenset[str]] = frozenset({"\u200c", "\u200d"})


def is_indic_text(text: str) -> bool:
    """Return whether *text* contains a code point from a supported Indic script."""

    return any(_is_indic_codepoint(ord(char)) for char in text)


def iter_grapheme_clusters(text: str) -> Iterator[tuple[int, int]]:
    """Yield extended grapheme-cluster boundaries as ``(start, end)`` pairs.

    The iterator implements the core rules from Unicode Standard Annex #29
    using only :mod:`unicodedata`. It additionally treats virama-linked Indic
    consonants as one akshara, including join controls, dependent vowels,
    nuktas, and Reph sequences across the nine supported Indic scripts.

    Args:
        text: Source Unicode text. Returned offsets index this exact string.

    Yields:
        Half-open ``(start, end)`` code-point offsets for each cluster.
    """

    if not text:
        return

    cluster_start = 0
    regional_indicators = 1 if _is_regional_indicator(text[0]) else 0

    for index in range(1, len(text)):
        if _has_grapheme_break(
            text,
            cluster_start=cluster_start,
            index=index,
            regional_indicators=regional_indicators,
        ):
            yield cluster_start, index
            cluster_start = index
            regional_indicators = 1 if _is_regional_indicator(text[index]) else 0
        elif _is_regional_indicator(text[index]):
            regional_indicators += 1
        else:
            regional_indicators = 0

    yield cluster_start, len(text)


def is_grapheme_boundary(index: int, text: str) -> bool:
    """Return whether *index* is a grapheme boundary in *text*."""

    if index < 0 or index > len(text):
        return False
    if index in {0, len(text)}:
        return True
    return any(end == index for _, end in iter_grapheme_clusters(text))


def snap_span_to_graphemes(start: int, end: int, text: str) -> tuple[int, int]:
    """Expand ``[start, end)`` to the nearest enclosing grapheme boundaries.

    Invalid offsets are clamped to *text*. An empty span already on a boundary
    remains empty; an empty span inside a cluster expands to that whole cluster.
    """

    safe_start = max(0, min(int(start), len(text)))
    safe_end = max(safe_start, min(int(end), len(text)))
    snapped_start = safe_start
    snapped_end = safe_end

    for cluster_start, cluster_end in iter_grapheme_clusters(text):
        if cluster_start < safe_start < cluster_end:
            snapped_start = cluster_start
        if cluster_start < safe_end < cluster_end:
            snapped_end = cluster_end
        if cluster_start >= safe_end:
            break

    return snapped_start, max(snapped_start, snapped_end)


def trim_span_whitespace(start: int, end: int, text: str) -> tuple[int, int]:
    """Strip leading and trailing whitespace from ``text[start:end]``.

    Returns the inclusive ``[start, end)`` indices into ``text`` after
    trimming. Input offsets are snapped outward before whole whitespace
    clusters are removed, so the returned offsets never split a grapheme.
    """

    start, end = snap_span_to_graphemes(start, end, text)
    clusters = list(iter_grapheme_clusters(text[start:end]))
    if not clusters:
        return start, end

    left = 0
    while left < len(clusters):
        cluster_start, cluster_end = clusters[left]
        if not text[start + cluster_start : start + cluster_end].isspace():
            break
        left += 1

    if left == len(clusters):
        return end, end

    right = len(clusters)
    while right > left:
        cluster_start, cluster_end = clusters[right - 1]
        if not text[start + cluster_start : start + cluster_end].isspace():
            break
        right -= 1

    trimmed_start = start + clusters[left][0]
    trimmed_end = start + clusters[right - 1][1]
    return trimmed_start, trimmed_end


_PRIVACY_FILTER_SPAN_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("url", re.compile(r"\b(?:https?://|www\.)[^\s,;)\]]+")),
    ("phone", re.compile(r"(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}")),
)


def refine_privacy_filter_span(
    label: str,
    start: int,
    end: int,
    text: str,
) -> tuple[int, int]:
    """Tighten obvious structured-PII spans when the model absorbs glue words.

    For email / URL / phone labels, locate the strict regex match inside
    the model-suggested span and shrink to that. For any label, drop a
    trailing ``" and"`` or ``" or"`` that the model often grabs because
    it sat next to the entity in training data.
    """
    start, end = trim_span_whitespace(start, end, text)
    span_text = text[start:end]
    normalized = label.lower()

    for label_hint, pattern in _PRIVACY_FILTER_SPAN_PATTERNS:
        if label_hint not in normalized:
            continue
        match = pattern.search(span_text)
        if match:
            return trim_span_whitespace(
                *snap_span_to_graphemes(
                    start + match.start(),
                    start + match.end(),
                    text,
                ),
                text,
            )

    for suffix in (" and", " or"):
        if span_text.lower().endswith(suffix):
            end -= len(suffix)
            break
    return trim_span_whitespace(start, end, text)


def _has_grapheme_break(
    text: str,
    *,
    cluster_start: int,
    index: int,
    regional_indicators: int,
) -> bool:
    previous = text[index - 1]
    current = text[index]

    # GB3, followed by GB4/GB5.
    if previous == "\r" and current == "\n":
        return False
    if _is_grapheme_control(previous) or _is_grapheme_control(current):
        return True

    # GB6-GB8: conjoining Hangul syllable sequences.
    previous_hangul = _hangul_type(previous)
    current_hangul = _hangul_type(current)
    if previous_hangul == "L" and current_hangul in {"L", "V", "LV", "LVT"}:
        return False
    if previous_hangul in {"LV", "V"} and current_hangul in {"V", "T"}:
        return False
    if previous_hangul in {"LVT", "T"} and current_hangul == "T":
        return False

    # GB9, GB9a, and GB9b.
    if _is_extend(current) or current == "\u200d" or _is_spacing_mark(current):
        return False
    if _is_prepend(previous):
        return False

    # GB9c plus explicit Indic aksara tailoring.
    if _is_indic_conjunct_boundary(text, cluster_start, index):
        return False

    # GB11: extended pictographic + Extend* + ZWJ + pictographic.
    if _is_extended_pictographic(current) and previous == "\u200d":
        cursor = index - 2
        while cursor >= cluster_start and _is_extend(text[cursor]):
            cursor -= 1
        if cursor >= cluster_start and _is_extended_pictographic(text[cursor]):
            return False

    # GB12/GB13: pair regional indicators from the start of the cluster.
    if (
        _is_regional_indicator(previous)
        and _is_regional_indicator(current)
        and regional_indicators % 2 == 1
    ):
        return False

    return True


def _is_indic_codepoint(codepoint: int) -> bool:
    return any(start <= codepoint <= end for start, end in _INDIC_SCRIPT_RANGES)


def _indic_script_index(char: str) -> int | None:
    codepoint = ord(char)
    for index, (start, end) in enumerate(_INDIC_SCRIPT_RANGES):
        if start <= codepoint <= end:
            return index
    return None


def _is_indic_consonant(char: str) -> bool:
    return _indic_script_index(char) is not None and unicodedata.category(
        char
    ).startswith("L")


def _is_indic_conjunct_boundary(
    text: str,
    cluster_start: int,
    index: int,
) -> bool:
    current = text[index]
    current_script = _indic_script_index(current)
    if current_script is None or not _is_indic_consonant(current):
        return False

    cursor = index - 1
    saw_virama = False
    while cursor >= cluster_start:
        char = text[cursor]
        if ord(char) in _INDIC_VIRAMAS:
            saw_virama = True
            cursor -= 1
            continue
        if _is_extend(char) or char in _JOIN_CONTROLS:
            cursor -= 1
            continue
        return (
            saw_virama
            and _is_indic_consonant(char)
            and _indic_script_index(char) == current_script
        )
    return False


def _is_extend(char: str) -> bool:
    codepoint = ord(char)
    return (
        unicodedata.category(char) in {"Mn", "Me"}
        or char == "\u200c"
        or 0x1F3FB <= codepoint <= 0x1F3FF
        or 0xFE00 <= codepoint <= 0xFE0F
        or 0xE0100 <= codepoint <= 0xE01EF
    )


def _is_spacing_mark(char: str) -> bool:
    return unicodedata.category(char) == "Mc"


def _is_grapheme_control(char: str) -> bool:
    if char in _JOIN_CONTROLS or _is_prepend(char):
        return False
    return unicodedata.category(char) in {"Cc", "Cf", "Cs", "Zl", "Zp"}


def _is_prepend(char: str) -> bool:
    codepoint = ord(char)
    return (
        0x0600 <= codepoint <= 0x0605
        or codepoint == 0x06DD
        or codepoint == 0x070F
        or codepoint == 0x0890
        or codepoint == 0x0891
        or codepoint == 0x08E2
        or codepoint == 0x110BD
        or codepoint == 0x110CD
    )


def _hangul_type(char: str) -> str | None:
    codepoint = ord(char)
    if 0x1100 <= codepoint <= 0x115F or 0xA960 <= codepoint <= 0xA97C:
        return "L"
    if 0x1160 <= codepoint <= 0x11A7 or 0xD7B0 <= codepoint <= 0xD7C6:
        return "V"
    if 0x11A8 <= codepoint <= 0x11FF or 0xD7CB <= codepoint <= 0xD7FB:
        return "T"
    if 0xAC00 <= codepoint <= 0xD7A3:
        return "LV" if (codepoint - 0xAC00) % 28 == 0 else "LVT"
    return None


def _is_regional_indicator(char: str) -> bool:
    return 0x1F1E6 <= ord(char) <= 0x1F1FF


def _is_extended_pictographic(char: str) -> bool:
    codepoint = ord(char)
    return (
        0x1F000 <= codepoint <= 0x1FAFF
        or 0x1FC00 <= codepoint <= 0x1FFFD
        or 0x2600 <= codepoint <= 0x27BF
        or codepoint in {0x00A9, 0x00AE, 0x203C, 0x2049, 0x2122, 0x2139}
    )


def _byte_offset(text: str, char_offset: int) -> int:
    return len(text[: max(0, char_offset)].encode("utf-8"))


def stable_span_id(label: str, start: int) -> str:
    """Return a deterministic PHI-free id for a streamed entity anchor."""
    digest = hashlib.sha256(f"{label}\0{int(start)}".encode("utf-8")).hexdigest()
    return f"ent_{digest[:16]}"


@dataclass(frozen=True)
class TokenClassificationSpan:
    """Entity span emitted by incremental token-classification streaming."""

    id: str
    label: str
    start: int
    end: int
    score: float
    text: str = ""
    byte_start: int | None = None
    byte_end: int | None = None

    def to_dict(self, *, include_text: bool = True) -> dict[str, object]:
        """Return a JSON-serializable span payload."""
        payload: dict[str, object] = {
            "id": self.id,
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "byte_start": self.byte_start,
            "byte_end": self.byte_end,
            "score": self.score,
        }
        if include_text:
            payload["text"] = self.text
        return payload

    def to_audit_dict(self) -> dict[str, object]:
        """Return a PHI-safe audit payload with hashes instead of raw text."""
        payload = self.to_dict(include_text=False)
        if self.text:
            payload["text_hash"] = (
                "sha256:" + hashlib.sha256(self.text.encode("utf-8")).hexdigest()
            )
        return payload


@dataclass(frozen=True)
class TokenClassificationStreamEvent:
    """Emit/retract/final event for streaming token classification."""

    type: str
    entity_id: str | None = None
    span: TokenClassificationSpan | None = None
    reason: str | None = None
    final_spans: tuple[TokenClassificationSpan, ...] = ()
    latency_ms: float | None = None
    window_chars: int | None = None

    def to_dict(self, *, include_text: bool = True) -> dict[str, object]:
        """Return a JSON-serializable event payload."""
        payload: dict[str, object] = {"type": self.type}
        if self.entity_id is not None:
            payload["entity_id"] = self.entity_id
        if self.reason is not None:
            payload["reason"] = self.reason
        if self.span is not None:
            payload["span"] = self.span.to_dict(include_text=include_text)
        if self.final_spans:
            payload["final_spans"] = [
                span.to_dict(include_text=include_text) for span in self.final_spans
            ]
        if self.latency_ms is not None:
            payload["latency_ms"] = self.latency_ms
        if self.window_chars is not None:
            payload["window_chars"] = self.window_chars
        return payload

    def to_audit_dict(self) -> dict[str, object]:
        """Return a PHI-safe event payload for logs and audit trails."""
        payload: dict[str, object] = {"type": self.type}
        if self.entity_id is not None:
            payload["entity_id"] = self.entity_id
        if self.reason is not None:
            payload["reason"] = self.reason
        if self.span is not None:
            payload["span"] = self.span.to_audit_dict()
        if self.final_spans:
            payload["final_spans"] = [span.to_audit_dict() for span in self.final_spans]
        if self.latency_ms is not None:
            payload["latency_ms"] = self.latency_ms
        if self.window_chars is not None:
            payload["window_chars"] = self.window_chars
        return payload


def coerce_token_classification_spans(
    predictions: list[object],
    text: str,
    *,
    base_offset: int = 0,
    base_byte_offset: int = 0,
    confidence_threshold: float = 0.0,
) -> list[TokenClassificationSpan]:
    """Convert backend entity dicts/objects to absolute streaming spans."""
    spans: list[TokenClassificationSpan] = []
    for item in predictions:
        if isinstance(item, TokenClassificationSpan):
            span = item
            if base_offset or base_byte_offset:
                span = replace(
                    span,
                    start=span.start + base_offset,
                    end=span.end + base_offset,
                    byte_start=(
                        None
                        if span.byte_start is None
                        else span.byte_start + base_byte_offset
                    ),
                    byte_end=(
                        None
                        if span.byte_end is None
                        else span.byte_end + base_byte_offset
                    ),
                )
            spans.append(span)
            continue

        getter = (
            item.get
            if isinstance(item, dict)
            else lambda key, default=None: getattr(item, key, default)
        )
        raw_start = getter("start")
        raw_end = getter("end")
        if raw_start is None or raw_end is None:
            continue
        start = int(raw_start)
        end = int(raw_end)
        if end <= start:
            continue
        score = float(
            getter(
                "score",
                getter("confidence", 0.0),
            )
            or 0.0
        )
        if score < confidence_threshold:
            continue
        label = str(
            getter(
                "entity_group",
                getter("entity", getter("label", getter("entity_type", "UNKNOWN"))),
            )
            or "UNKNOWN"
        )
        label = (
            label.removeprefix("B-")
            .removeprefix("I-")
            .removeprefix("E-")
            .removeprefix("S-")
        )
        local_text = str(getter("word", getter("text", text[start:end])) or "")
        absolute_start = base_offset + start
        absolute_end = base_offset + end
        byte_start = base_byte_offset + _byte_offset(text, start)
        byte_end = base_byte_offset + _byte_offset(text, end)
        spans.append(
            TokenClassificationSpan(
                id=stable_span_id(label, absolute_start),
                label=label,
                start=absolute_start,
                end=absolute_end,
                byte_start=byte_start,
                byte_end=byte_end,
                score=score,
                text=local_text or text[start:end],
            )
        )

    return sorted(spans, key=lambda span: (span.start, span.end, span.label, span.id))


def reconcile_stream_spans(
    active_spans: dict[str, TokenClassificationSpan],
    current_spans: list[TokenClassificationSpan],
) -> tuple[list[TokenClassificationStreamEvent], dict[str, TokenClassificationSpan]]:
    """Compute retract/emit events needed to reach ``current_spans``."""
    events: list[TokenClassificationStreamEvent] = []
    next_active = {span.id: span for span in current_spans}

    for entity_id, previous in sorted(
        active_spans.items(), key=lambda item: (item[1].start, item[1].end, item[0])
    ):
        current = next_active.get(entity_id)
        if current is None:
            events.append(
                TokenClassificationStreamEvent(
                    type="retract",
                    entity_id=entity_id,
                    span=previous,
                    reason="span_removed",
                )
            )
        elif _span_changed(previous, current):
            events.append(
                TokenClassificationStreamEvent(
                    type="retract",
                    entity_id=entity_id,
                    span=previous,
                    reason="span_updated",
                )
            )

    for span in current_spans:
        previous = active_spans.get(span.id)
        if previous is None or _span_changed(previous, span):
            events.append(
                TokenClassificationStreamEvent(
                    type="emit",
                    entity_id=span.id,
                    span=span,
                )
            )

    return events, next_active


def _span_changed(
    previous: TokenClassificationSpan,
    current: TokenClassificationSpan,
) -> bool:
    return (
        previous.label != current.label
        or previous.start != current.start
        or previous.end != current.end
        or previous.byte_start != current.byte_start
        or previous.byte_end != current.byte_end
        or previous.text != current.text
    )


def stable_span_key(span: Any) -> tuple[int, int, str, str]:
    """Return a deterministic ordering key for span-like objects.

    The key intentionally depends only on source offsets plus optional label and
    text fields, so downstream decoders can make stable tie-break decisions
    without depending on object identity or model output order.
    """

    start = int(getattr(span, "start", 0))
    end = int(getattr(span, "end", start))
    label = str(getattr(span, "label", ""))
    span_text = str(getattr(span, "text", ""))
    return start, end, label.casefold(), span_text.casefold()


__all__ = [
    "TokenClassificationSpan",
    "TokenClassificationStreamEvent",
    "coerce_token_classification_spans",
    "reconcile_stream_spans",
    "refine_privacy_filter_span",
    "stable_span_id",
    "stable_span_key",
    "trim_span_whitespace",
]
