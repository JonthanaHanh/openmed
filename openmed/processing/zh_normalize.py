"""Full-width/half-width normalization with offset preservation (roadmap v1.9).

Chinese documents freely mix full-width Latin letters, digits, and punctuation
(``ＡＢＣ１２３，：``) with half-width ASCII, so a phone number or ID written in
full-width digits never matches the half-width-oriented PHI regexes or
tokenizers. Blunt NFKC normalization fixes matching but can change string length
and merge characters, silently shifting every downstream PHI offset.

``normalize_width`` maps full-width variants to half-width while leaving genuine
full-width CJK content untouched, and returns an explicit per-character offset
alignment so a span detected on the normalized text maps back to the exact
original code points. The mapping is 1:1 in the common case; the rare
compatibility expansions (``㎏`` -> ``kg``) are handled by recording, for every
normalized character, the original code point it came from.

Two conventions are supported: the recommended CJK convention (Latin, digits,
and symbols to half-width; everything else, including Han, left as-is) and strict
per-character NFKC.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date

#: Recommended CJK convention: full-width ASCII/space to half-width, Han as-is.
CJK_CONVENTION = "cjk"
#: Strict per-character NFKC normalization.
STRICT_NFKC = "nfkc"
WIDTH_CONVENTIONS = (CJK_CONVENTION, STRICT_NFKC)

# Full-width ASCII variants U+FF01-FF5E map to U+0021-007E by this offset.
_FULLWIDTH_START = 0xFF01
_FULLWIDTH_END = 0xFF5E
_WIDTH_OFFSET = 0xFEE0
#: Ideographic space, normalized to a configurable target.
IDEOGRAPHIC_SPACE = "　"

# Chinese numeral characters intentionally exclude colloquial variants such as
# 两: the parser covers the everyday and financial forms in the public contract.
CHINESE_NUMERAL_CHARACTERS = (
    "〇零一二三四五六七八九十百千万亿壹贰叁肆伍陆柒捌玖拾佰仟萬億"
)
CHINESE_NUMERAL_PATTERN = rf"[{CHINESE_NUMERAL_CHARACTERS}]+"

_DIGIT_VALUES = {
    "〇": 0,
    "零": 0,
    "一": 1,
    "壹": 1,
    "二": 2,
    "贰": 2,
    "三": 3,
    "叁": 3,
    "四": 4,
    "肆": 4,
    "五": 5,
    "伍": 5,
    "六": 6,
    "陆": 6,
    "七": 7,
    "柒": 7,
    "八": 8,
    "捌": 8,
    "九": 9,
    "玖": 9,
}
_SMALL_UNIT_VALUES = {
    "十": 10,
    "拾": 10,
    "百": 100,
    "佰": 100,
    "千": 1000,
    "仟": 1000,
}
_LARGE_UNIT_VALUES = {
    "万": 10_000,
    "萬": 10_000,
    "亿": 100_000_000,
    "億": 100_000_000,
}
_ALL_NUMERAL_VALUES = _DIGIT_VALUES | _SMALL_UNIT_VALUES | _LARGE_UNIT_VALUES
_CHINESE_NUMERAL_RE = re.compile(CHINESE_NUMERAL_PATTERN)
_CHINESE_DATE_RE = re.compile(
    rf"(?P<year>{CHINESE_NUMERAL_PATTERN})\s*年\s*"
    rf"(?P<month>{CHINESE_NUMERAL_PATTERN})\s*月\s*"
    rf"(?P<day>{CHINESE_NUMERAL_PATTERN})\s*日"
)


@dataclass(frozen=True)
class ChineseNumberSpan:
    """A parsed Chinese numeral and its exact source code-point span."""

    value: int
    start: int
    end: int
    text: str

    @property
    def span(self) -> tuple[int, int]:
        """Return the half-open ``(start, end)`` source span."""

        return self.start, self.end

    @property
    def source_span(self) -> tuple[int, int]:
        """Return the half-open source span as an explicit alias."""

        return self.span


@dataclass(frozen=True)
class ChineseDateNormalization:
    """A Chinese year/month/day expression with source-aligned number runs."""

    year: int
    month: int
    day: int
    start: int
    end: int
    text: str
    number_spans: tuple[ChineseNumberSpan, ChineseNumberSpan, ChineseNumberSpan]

    @property
    def span(self) -> tuple[int, int]:
        """Return the half-open span covering the complete source date."""

        return self.start, self.end

    @property
    def source_span(self) -> tuple[int, int]:
        """Return the complete source date span as an explicit alias."""

        return self.span

    @property
    def normalized(self) -> str:
        """Return the parsed date in ISO ``YYYY-MM-DD`` form."""

        return f"{self.year:04d}-{self.month:02d}-{self.day:02d}"


@dataclass(frozen=True)
class WidthNormalization:
    """Width-normalized text with an offset alignment to the original.

    ``char_origins[i]`` is the ``(start, end)`` code-point span in ``original``
    that normalized character ``i`` was produced from.
    """

    text: str
    original: str
    char_origins: tuple[tuple[int, int], ...]

    def to_original_span(self, start: int, end: int) -> tuple[int, int]:
        """Map a ``[start, end)`` span on the normalized text to the original.

        The returned span is expressed in original code-point offsets and
        covers exactly the source characters the normalized span came from.
        """

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
        return self.char_origins[start][0], self.char_origins[end - 1][1]


def _parse_below_ten_thousand(numeral: str) -> int:
    if not numeral:
        return 0

    if not any(char in _SMALL_UNIT_VALUES for char in numeral):
        if not all(char in _DIGIT_VALUES for char in numeral):
            raise ValueError(f"invalid Chinese numeral section: {numeral!r}")
        value = 0
        for char in numeral:
            value = value * 10 + _DIGIT_VALUES[char]
        return value

    total = 0
    pending_digit: int | None = None
    last_unit = 10_000
    zero_placeholder = False

    for char in numeral:
        if char in _DIGIT_VALUES:
            digit = _DIGIT_VALUES[char]
            if digit == 0:
                pending_digit = None
                zero_placeholder = True
                continue
            if pending_digit is not None:
                raise ValueError(f"adjacent digits in unit numeral: {numeral!r}")
            pending_digit = digit
            zero_placeholder = False
            continue

        unit = _SMALL_UNIT_VALUES.get(char)
        if unit is None or unit >= last_unit:
            raise ValueError(f"invalid unit order in Chinese numeral: {numeral!r}")
        if pending_digit is None and zero_placeholder:
            raise ValueError(f"unit cannot follow a zero placeholder: {numeral!r}")

        total += (pending_digit if pending_digit is not None else 1) * unit
        pending_digit = None
        zero_placeholder = False
        last_unit = unit

    if pending_digit is not None:
        total += pending_digit
    return total


def _parse_below_hundred_million(numeral: str) -> int:
    positions = [
        index
        for index, char in enumerate(numeral)
        if _LARGE_UNIT_VALUES.get(char) == 10_000
    ]
    if len(positions) > 1:
        raise ValueError(f"repeated ten-thousand unit in numeral: {numeral!r}")
    if not positions:
        return _parse_below_ten_thousand(numeral)

    position = positions[0]
    left = numeral[:position]
    right = numeral[position + 1 :]
    left_value = _parse_below_ten_thousand(left) if left else 1
    return left_value * 10_000 + _parse_below_ten_thousand(right)


def parse_chinese_numeral(numeral: str) -> int:
    """Parse an everyday or financial-form Chinese numeral as an integer.

    Digit-only forms such as ``二〇二四`` are read positionally. Unit forms use
    descending ``十/百/千`` multipliers inside ``万/亿`` sections, so interior
    zero placeholders do not contribute a digit of their own. A bare zero is
    rejected because it is not a useful numeral span for PHI detection.

    Args:
        numeral: A non-empty run of supported Chinese numeral characters.

    Returns:
        The integer represented by ``numeral``.

    Raises:
        TypeError: If ``numeral`` is not a string.
        ValueError: If the run is empty, unsupported, structurally invalid, or
            consists only of zero characters.
    """

    if not isinstance(numeral, str):
        raise TypeError("numeral must be a string")
    if not numeral:
        raise ValueError("numeral must not be empty")
    unsupported = set(numeral) - _ALL_NUMERAL_VALUES.keys()
    if unsupported:
        raise ValueError(f"unsupported Chinese numeral characters: {unsupported!r}")
    if not any(_ALL_NUMERAL_VALUES[char] > 0 for char in numeral):
        raise ValueError("a standalone Chinese zero is not a numeral span")

    hundred_million_positions = [
        index
        for index, char in enumerate(numeral)
        if _LARGE_UNIT_VALUES.get(char) == 100_000_000
    ]
    if len(hundred_million_positions) > 1:
        raise ValueError(f"repeated hundred-million unit in numeral: {numeral!r}")
    if not hundred_million_positions:
        return _parse_below_hundred_million(numeral)

    position = hundred_million_positions[0]
    left = numeral[:position]
    right = numeral[position + 1 :]
    left_value = _parse_below_hundred_million(left) if left else 1
    return left_value * 100_000_000 + _parse_below_hundred_million(right)


def find_chinese_numbers(text: str) -> list[ChineseNumberSpan]:
    """Find valid Chinese numeral runs with values and source offsets.

    Offsets are Python string indices and therefore exact Unicode code-point
    offsets. Standalone ``〇`` or ``零`` runs and structurally invalid runs are
    ignored rather than emitted as ambiguous numeric spans.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    numbers: list[ChineseNumberSpan] = []
    for match in _CHINESE_NUMERAL_RE.finditer(text):
        try:
            value = parse_chinese_numeral(match.group())
        except ValueError:
            continue
        numbers.append(
            ChineseNumberSpan(
                value=value,
                start=match.start(),
                end=match.end(),
                text=match.group(),
            )
        )
    return numbers


def normalize_chinese_dates(text: str) -> list[ChineseDateNormalization]:
    """Recognize Chinese year/month/day dates without losing source offsets.

    Each result contains the parsed ISO-compatible fields, the whole date span,
    and three :class:`ChineseNumberSpan` records for the year, month, and day.
    Invalid calendar dates are ignored.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    dates: list[ChineseDateNormalization] = []
    for match in _CHINESE_DATE_RE.finditer(text):
        parts: list[ChineseNumberSpan] = []
        values: list[int] = []
        try:
            for name in ("year", "month", "day"):
                start, end = match.span(name)
                value = parse_chinese_numeral(match.group(name))
                values.append(value)
                parts.append(
                    ChineseNumberSpan(
                        value=value,
                        start=start,
                        end=end,
                        text=text[start:end],
                    )
                )
            date(*values)
        except ValueError:
            continue

        dates.append(
            ChineseDateNormalization(
                year=values[0],
                month=values[1],
                day=values[2],
                start=match.start(),
                end=match.end(),
                text=match.group(),
                number_spans=(parts[0], parts[1], parts[2]),
            )
        )
    return dates


def _convert_char(ch: str, convention: str, space_target: str) -> str:
    if convention == CJK_CONVENTION:
        code_point = ord(ch)
        if _FULLWIDTH_START <= code_point <= _FULLWIDTH_END:
            return chr(code_point - _WIDTH_OFFSET)
        if ch == IDEOGRAPHIC_SPACE:
            return space_target
        return ch
    if convention == STRICT_NFKC:
        return unicodedata.normalize("NFKC", ch)
    raise ValueError(f"unknown width convention: {convention!r}")


def normalize_width(
    text: str,
    *,
    convention: str = CJK_CONVENTION,
    space_target: str = " ",
) -> WidthNormalization:
    """Normalize full-width characters to half-width, preserving offsets.

    ``convention`` selects the recommended :data:`CJK_CONVENTION` (default) or
    strict :data:`STRICT_NFKC`. ``space_target`` is the replacement for the
    ideographic space under the CJK convention. Returns a
    :class:`WidthNormalization` whose ``char_origins`` map every normalized
    character back to its source code points.
    """

    if convention not in WIDTH_CONVENTIONS:
        raise ValueError(f"unknown width convention: {convention!r}")

    chars: list[str] = []
    origins: list[tuple[int, int]] = []
    for index, char in enumerate(text):
        for normalized_char in _convert_char(char, convention, space_target):
            chars.append(normalized_char)
            origins.append((index, index + 1))

    return WidthNormalization(
        text="".join(chars),
        original=text,
        char_origins=tuple(origins),
    )


def detect_width_normalized(
    text: str,
    matcher: Callable[[str], Iterable[Sequence[object]]],
    *,
    convention: str = CJK_CONVENTION,
    space_target: str = " ",
) -> list[tuple[object, ...]]:
    """Run ``matcher`` on width-normalized ``text`` and map spans back.

    This is the early width pre-pass: half-width-oriented PHI patterns match
    full-width phone/ID inputs once the text is normalized. ``matcher`` takes
    the normalized text and returns an iterable of items whose first two
    elements are the ``(start, end)`` span on the normalized text; any trailing
    elements (label, score, ...) are preserved. Each returned tuple is
    ``(original_start, original_end, *trailing)`` in original code-point offsets.
    """

    normalization = normalize_width(
        text, convention=convention, space_target=space_target
    )
    results: list[tuple[object, ...]] = []
    for item in matcher(normalization.text):
        start = int(item[0])  # type: ignore[call-overload]
        end = int(item[1])  # type: ignore[call-overload]
        original_start, original_end = normalization.to_original_span(start, end)
        results.append((original_start, original_end, *tuple(item[2:])))
    return results


__all__ = [
    "CJK_CONVENTION",
    "STRICT_NFKC",
    "WIDTH_CONVENTIONS",
    "IDEOGRAPHIC_SPACE",
    "CHINESE_NUMERAL_CHARACTERS",
    "CHINESE_NUMERAL_PATTERN",
    "ChineseNumberSpan",
    "ChineseDateNormalization",
    "parse_chinese_numeral",
    "find_chinese_numbers",
    "normalize_chinese_dates",
    "WidthNormalization",
    "normalize_width",
    "detect_width_normalized",
]
