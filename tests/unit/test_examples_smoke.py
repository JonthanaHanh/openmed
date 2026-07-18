"""Smoke tests for import-safe example scripts."""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples import (
    clinical_ner_families,
    gradio_deid_app,
    onboarding_india_dpdp,
)


def test_clinical_ner_families_example_is_syntactically_valid():
    source = Path("examples/clinical_ner_families.py").read_text(encoding="utf-8")

    ast.parse(source)


def test_clinical_ner_families_selects_three_registry_families():
    assert {"Disease", "Pharmaceutical", "Oncology"}.issubset(
        clinical_ner_families.NER_FAMILIES
    )

    selections = clinical_ner_families.selected_families()

    assert {selection.family for selection in selections} == {
        "Disease",
        "Pharmaceutical",
        "Oncology",
    }
    assert all(selection.source == "accurate tier" for selection in selections)
    assert all(
        selection.model.model_id.startswith("OpenMed/") for selection in selections
    )


def test_clinical_ner_families_uses_mocked_analyzer_without_network(
    capsys,
    monkeypatch,
):
    monkeypatch.delenv(clinical_ner_families.ALLOW_DOWNLOAD_ENV, raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    calls = []

    def fake_analyzer(text, *, model_name, confidence_threshold, group_entities):
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
        calls.append(
            {
                "text": text,
                "model_name": model_name,
                "confidence_threshold": confidence_threshold,
                "group_entities": group_entities,
            }
        )
        return SimpleNamespace(
            entities=[
                SimpleNamespace(
                    label="MOCK_ENTITY",
                    text="synthetic span",
                    confidence=0.99,
                )
            ]
        )

    clinical_ner_families.run_family_extraction(analyzer=fake_analyzer)

    assert len(calls) == 3
    assert all(call["group_entities"] is True for call in calls)
    assert all(call["model_name"].startswith("OpenMed/") for call in calls)
    assert "HF_HUB_OFFLINE" not in os.environ
    assert "TRANSFORMERS_OFFLINE" not in os.environ
    captured = capsys.readouterr().out
    assert "Disease" in captured
    assert "Pharmaceutical" in captured
    assert "Oncology" in captured
    assert "MOCK_ENTITY" in captured


def test_clinical_ner_families_reports_offline_unavailable(capsys):
    def unavailable_analyzer(*args, **kwargs):
        raise OSError("cached files not found")

    clinical_ner_families.run_family_extraction(
        families=("Disease",),
        analyzer=unavailable_analyzer,
    )

    captured = capsys.readouterr().out
    assert "Disease: model unavailable offline" in captured
    assert "cached files not found" in captured


def test_gradio_deid_app_is_syntactically_valid():
    source = Path("examples/gradio_deid_app.py").read_text(encoding="utf-8")

    ast.parse(source)


def test_gradio_deid_app_exposes_public_surface():
    assert gradio_deid_app.DEIDENTIFICATION_METHODS == ("mask", "replace", "hash")
    assert callable(gradio_deid_app.build_demo)
    assert callable(gradio_deid_app.run_deidentification)
    assert "Synthetic note" in gradio_deid_app.SYNTHETIC_CLINICAL_TEXT


def test_gradio_deid_app_builds_entity_rows():
    entities = [
        SimpleNamespace(label="NAME", text="John Doe", start=0, end=8, confidence=0.97),
        SimpleNamespace(
            label="EMAIL", text="j@x.org", start=20, end=27, confidence=0.5
        ),
    ]

    rows = gradio_deid_app.entities_to_rows(entities)

    assert rows == [
        ["NAME", "John Doe", "0", "8", "0.97"],
        ["EMAIL", "j@x.org", "20", "27", "0.50"],
    ]


def test_gradio_deid_app_run_uses_deidentify_without_network(monkeypatch):
    calls = []

    def fake_deidentify(text, *, method):
        calls.append((text, method))
        return SimpleNamespace(
            deidentified_text="[NAME] was seen.",
            pii_entities=[
                SimpleNamespace(
                    label="NAME", text="John Doe", start=0, end=8, confidence=0.9
                )
            ],
        )

    monkeypatch.setattr(gradio_deid_app, "deidentify", fake_deidentify)

    view = gradio_deid_app.run_deidentification("John Doe was seen.", "mask")

    assert calls == [("John Doe was seen.", "mask")]
    assert view.deidentified_text == "[NAME] was seen."
    assert view.entity_rows == [["NAME", "John Doe", "0", "8", "0.90"]]


def test_gradio_deid_app_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unsupported method"):
        gradio_deid_app.run_deidentification("text", "encrypt")


def test_gradio_deid_app_missing_gradio_prints_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "gradio", None)

    with pytest.raises(SystemExit) as excinfo:
        gradio_deid_app.build_demo()

    assert "pip install gradio" in str(excinfo.value)


def test_onboarding_india_dpdp_example_is_syntactically_valid():
    source = Path("examples/onboarding_india_dpdp.py").read_text(encoding="utf-8")

    ast.parse(source)


def test_onboarding_india_dpdp_recognizer_covers_synthetic_direct_identifiers():
    from openmed.core.custom_recognizer import CustomRecognizer

    recognizer = CustomRecognizer.from_config(
        onboarding_india_dpdp.INDIA_CUSTOM_RECOGNIZER
    )

    assert [
        (entity.text, entity.label)
        for entity in recognizer.detect_entities(
            onboarding_india_dpdp.SYNTHETIC_HINGLISH_NOTE
        )
    ] == [
        (onboarding_india_dpdp.SYNTHETIC_PERSON, "PERSON"),
        (onboarding_india_dpdp.SYNTHETIC_AADHAAR, "ID_NUM"),
        (onboarding_india_dpdp.SYNTHETIC_ABHA, "ID_NUM"),
    ]


def test_onboarding_india_dpdp_runs_policy_pipeline_without_network(monkeypatch):
    from openmed.core import pii
    from openmed.processing.outputs import PredictionResult

    calls = []

    def fake_extract_pii(text, model_name, *args, **kwargs):
        calls.append(
            {
                "text": text,
                "model_name": model_name,
                "lang": kwargs["lang"],
            }
        )
        return PredictionResult(
            text=text,
            entities=[],
            model_name=model_name,
            timestamp="2026-01-01T00:00:00",
        )

    monkeypatch.setattr(pii, "extract_pii", fake_extract_pii)

    result = onboarding_india_dpdp.run()

    assert calls == [
        {
            "text": onboarding_india_dpdp.SYNTHETIC_HINGLISH_NOTE,
            "model_name": onboarding_india_dpdp.HINDI_MODEL_ID,
            "lang": "hi",
        }
    ]
    onboarding_india_dpdp.assert_synthetic_pii_is_masked(result.deidentified_text)
    protected_values = {entity.text for entity in result.pii_entities}
    assert {
        onboarding_india_dpdp.SYNTHETIC_AADHAAR,
        onboarding_india_dpdp.SYNTHETIC_ABHA,
    }.issubset(protected_values)
    assert "[PERSON]" in result.deidentified_text
