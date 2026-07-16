from hypothesis import given
from hypothesis import strategies as st

from openmed.core.pii_i18n import NATIONAL_ID_ONLY_LANGUAGES, SUPPORTED_LANGUAGES
from openmed.core.script_detect import (
    SCRIPT_LANGUAGE_HINTS,
    SUPPORTED_SCRIPTS,
    UNKNOWN_SCRIPT,
    candidate_languages_for_script,
    detect_script,
    normalize_for_pii_detection,
    segment_by_script,
)


def _assert_offsets_cover_text(
    segments: list[tuple[int, int, str]],
    text: str,
) -> None:
    cursor = 0
    for start, end, script in segments:
        assert script
        assert start == cursor
        assert start < end
        assert text[start:end]
        cursor = end
    assert cursor == len(text)


def test_detect_script_classifies_single_script_samples():
    samples = {
        "Patient John Smith": "Latin",
        "المريض أحمد علي": "Arabic",
        "ታካሚ ሰላም ተስፋዬ": "Ethiopic",
        "患者 佐藤花子": "Han",
        "かな カタカナ": "Hiragana/Katakana",
        "환자 김민수": "Hangul",
        "Пациент Иван": "Cyrillic",
        "मरीज़ अनिता शर्मा": "Devanagari",
        "రోగి సీత రెడ్డి": "Telugu",
        "Ασθενής Νίκος": "Greek",
        "מטופל דוד כהן": "Hebrew",
        "ผู้ป่วย สมชาย": "Thai",
    }

    for text, script in samples.items():
        assert detect_script(text) == script


def test_detect_script_ignores_neutral_characters():
    assert detect_script("  MRN-12345  ") == "Latin"
    assert detect_script("12345 / --") == UNKNOWN_SCRIPT


def test_segment_by_script_mixed_latin_arabic_offsets_cover_text():
    text = "Patient Ahmad راجع العيادة 5mg"
    segments = list(segment_by_script(text))

    _assert_offsets_cover_text(segments, text)
    assert [script for _, _, script in segments] == ["Latin", "Arabic", "Latin"]
    assert "".join(text[start:end] for start, end, _ in segments) == text


def test_segment_by_script_mixed_latin_han_offsets_cover_text():
    text = "MRN 42 患者 佐藤 visited"
    segments = list(segment_by_script(text))

    _assert_offsets_cover_text(segments, text)
    assert [script for _, _, script in segments] == ["Latin", "Han", "Latin"]
    assert "".join(text[start:end] for start, end, _ in segments) == text


def test_detect_script_covers_all_ethiopic_unicode_blocks():
    samples = ("ሀ", "ᎀ", "ⶀ", "ꬁ", "𞟠")

    assert all(detect_script(char) == "Ethiopic" for char in samples)


def test_segment_by_script_mixed_amharic_latin_has_exact_offsets():
    text = "ታካሚ Selam፡ ቀጠሮ"

    assert list(segment_by_script(text)) == [
        (0, 4, "Ethiopic"),
        (4, 11, "Latin"),
        (11, 14, "Ethiopic"),
    ]


def test_script_language_hints_cover_detectable_scripts():
    expected_scripts = set(SUPPORTED_SCRIPTS) | {UNKNOWN_SCRIPT}

    assert expected_scripts <= set(SCRIPT_LANGUAGE_HINTS)
    for script in expected_scripts:
        hints = candidate_languages_for_script(script)
        assert hints
        assert set(hints) <= SUPPORTED_LANGUAGES | NATIONAL_ID_ONLY_LANGUAGES


def test_normalize_for_pii_detection_folds_obfuscation_with_offset_map():
    text = "Patient J\u200bo\u0301hn D\u03bfe"
    normalized = normalize_for_pii_detection(text)

    assert normalized.text == "Patient John Doe"
    assert normalized.changed
    assert normalized.mixed_script
    assert normalized.removed_zero_width == 1
    assert normalized.stripped_combining_marks == 1
    assert normalized.folded_confusables == 1
    assert normalized.remap_span(8, 16) == (8, len(text))
    assert "Patient" not in normalized.to_metadata()


def test_normalize_for_pii_detection_strips_standalone_ethiopic_mark():
    normalized = normalize_for_pii_detection("\u135f")

    assert normalized.text == ""
    assert normalized.stripped_combining_marks == 1


@given(
    before=st.lists(st.sampled_from(tuple("ሀለሐመሠረሰቀበተነአከወዘየደገጠጸፈፐ")), min_size=1),
    after=st.lists(st.sampled_from(tuple("ሀለሐመሠረሰቀበተነአከወዘየደገጠጸፈፐ")), max_size=8),
    prefix=st.sampled_from(("", "ስም፡ ", "Patient ")),
    suffix=st.sampled_from(("", "።", " visited")),
)
def test_ethiopic_combining_mark_remaps_without_offset_drift(
    before: list[str],
    after: list[str],
    prefix: str,
    suffix: str,
):
    marked_value = f"{''.join(before)}\u135f{''.join(after)}"
    text = f"{prefix}{marked_value}{suffix}"
    normalized = normalize_for_pii_detection(text)
    value_start = len(prefix)
    value_end = value_start + len(marked_value)
    grapheme_start = value_start + len(before) - 1
    grapheme_end = grapheme_start + 2

    assert normalized.text == text
    assert normalized.stripped_combining_marks == 0
    assert normalized.remap_span(value_start, value_end) == (value_start, value_end)
    assert normalized.remap_span(grapheme_start, grapheme_end) == (
        grapheme_start,
        grapheme_end,
    )
    assert text[grapheme_start:grapheme_end].endswith("\u135f")
