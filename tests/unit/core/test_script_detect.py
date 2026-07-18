import unicodedata

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
    snap_span_to_grapheme_boundaries,
)


@st.composite
def _yoruba_cluster_spans(draw):
    clusters = draw(
        st.lists(
            st.sampled_from(("ọ́", "ẹ̀", "ṣ", "á", "ì", "ń", "ǹ")),
            min_size=1,
            max_size=12,
        )
    )
    use_nfd = draw(
        st.lists(
            st.booleans(),
            min_size=len(clusters),
            max_size=len(clusters),
        )
    )
    rendered = [
        unicodedata.normalize("NFD" if decomposed else "NFC", cluster)
        for cluster, decomposed in zip(clusters, use_nfd)
    ]
    span_start = draw(st.integers(min_value=0, max_value=len(rendered) - 1))
    span_end = draw(st.integers(min_value=span_start + 1, max_value=len(rendered)))
    return rendered, span_start, span_end


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


@given(case=_yoruba_cluster_spans())
def test_yoruba_normalization_remaps_whole_nfc_nfd_and_mixed_clusters(case):
    clusters, selected_start, selected_end = case
    prefix = unicodedata.normalize("NFD", "Àkọsílẹ̀: ")
    suffix = unicodedata.normalize("NFC", " parí.")
    before = "".join(clusters[:selected_start])
    selected = "".join(clusters[selected_start:selected_end])
    text = prefix + "".join(clusters) + suffix

    normalized = normalize_for_pii_detection(text)
    normalized_start = len(normalize_for_pii_detection(prefix + before).text)
    normalized_end = len(normalize_for_pii_detection(prefix + before + selected).text)
    original_start, original_end = normalized.remap_span(
        normalized_start,
        normalized_end,
    )

    assert text[original_start:original_end] == selected
    assert not unicodedata.category(text[original_start]).startswith("M")
    assert original_end == len(text) or not unicodedata.category(
        text[original_end]
    ).startswith("M")


def test_snap_span_expands_both_sides_of_decomposed_yoruba_clusters():
    text = unicodedata.normalize("NFD", "Bọ́láńlé Adébáyọ̀")
    expected_start = text.index(unicodedata.normalize("NFD", "ọ́"))
    expected_end = len(text)
    inside_first_marks = expected_start + 1
    before_final_marks = expected_end - 2

    assert snap_span_to_grapheme_boundaries(
        text,
        inside_first_marks,
        before_final_marks,
    ) == (expected_start, expected_end)
