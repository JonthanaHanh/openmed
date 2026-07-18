"""Smoke tests for import-safe example scripts."""

from __future__ import annotations

import ast
import importlib
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from jsonschema import validate

from examples import clinical_ner_families, gradio_deid_app

deid_demo = importlib.import_module("examples.spaces.deid_demo.app")


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


@pytest.mark.parametrize(
    ("text", "expected_language"),
    [
        ("患者李晓雯今天复诊。", "zh"),
        ("रोगी आज जाँच के लिए आए।", "hi"),
        ("Patient Alex Example was seen today.", "en"),
        ("Patient Ananya ka follow-up aaj hai.", "en"),
        ("1234 / +1 555 0100", "en"),
    ],
)
def test_deid_demo_detects_unicode_script(text, expected_language):
    assert deid_demo.detect_script(text) == expected_language


@pytest.mark.parametrize("override", ["zh", "hi", "en"])
def test_deid_demo_manual_override_wins(override):
    route = deid_demo.resolve_model_route("患者 हिन्दी Patient", override)

    assert route.language == override
    assert route.model_id == deid_demo.MODEL_IDS[override]


@pytest.mark.parametrize(
    ("sample_language", "openmed_language"),
    [("zh", "en"), ("hi", "hi"), ("en", "en")],
)
def test_deid_demo_routes_samples_without_network(
    sample_language,
    openmed_language,
):
    calls = []

    def fake_deidentify(text, **kwargs):
        calls.append((text, kwargs))
        return SimpleNamespace(deidentified_text=f"[{sample_language.upper()} MASKED]")

    sample = deid_demo.SAMPLES[sample_language]
    result = deid_demo.run_deidentification(
        sample.text,
        deidentifier=fake_deidentify,
    )

    assert result.deidentified_text == f"[{sample_language.upper()} MASKED]"
    assert result.route.language == sample_language
    assert calls[0][0] == sample.text
    assert calls[0][1] == {
        "method": "mask",
        "model_name": deid_demo.MODEL_IDS[sample_language],
        "lang": openmed_language,
        "policy": "strict_no_leak",
        "use_safety_sweep": True,
        "custom_recognizer": deid_demo._sample_recognizer(sample_language),
        "keep_mapping": False,
        "audit": False,
        "cache_results": False,
    }


def test_deid_demo_rejects_unknown_override():
    with pytest.raises(ValueError, match="Unsupported language override"):
        deid_demo.resolve_model_route("Synthetic note", "fr")


def test_deid_demo_does_not_log_or_persist_input(caplog, monkeypatch, tmp_path):
    raw_input = "Synthetic secret ZH-709-NEVER-LOG"
    monkeypatch.chdir(tmp_path)

    def fake_deidentify(text, **kwargs):
        del text, kwargs
        return SimpleNamespace(deidentified_text="Synthetic secret [ID_NUM]")

    result = deid_demo.run_deidentification(
        raw_input,
        language_override="en",
        deidentifier=fake_deidentify,
    )

    assert result.deidentified_text == "Synthetic secret [ID_NUM]"
    assert raw_input not in caplog.text
    assert list(tmp_path.iterdir()) == []


def test_deid_demo_disclaimer_contains_all_locales():
    assert set(deid_demo.DISCLAIMERS) == {"zh", "hi", "en"}
    assert all(
        disclaimer in deid_demo.DISCLAIMER_MARKDOWN
        for disclaimer in deid_demo.DISCLAIMERS.values()
    )
    assert "Never paste real patient data" in deid_demo.DISCLAIMERS["en"]
    assert "真实患者数据" in deid_demo.DISCLAIMERS["zh"]
    assert "वास्तविक रोगी डेटा" in deid_demo.DISCLAIMERS["hi"]


def test_deid_demo_space_frontmatter_matches_schema():
    readme = Path("examples/spaces/deid_demo/README.md").read_text(encoding="utf-8")
    match = re.match(r"\A---\n(.*?)\n---\n", readme, flags=re.DOTALL)
    assert match is not None
    metadata = yaml.safe_load(match.group(1))
    schema = {
        "type": "object",
        "required": ["title", "sdk", "sdk_version", "app_file", "models"],
        "properties": {
            "title": {"type": "string", "minLength": 1},
            "sdk": {"const": "gradio"},
            "sdk_version": {"type": "string", "pattern": r"^\d+\.\d+\.\d+$"},
            "python_version": {"type": "string", "pattern": r"^3\.\d+(?:\.\d+)?$"},
            "app_file": {"const": "app.py"},
            "models": {
                "type": "array",
                "minItems": 3,
                "items": {"type": "string", "pattern": r"^OpenMed/"},
            },
        },
    }

    validate(instance=metadata, schema=schema)
    assert Path("examples/spaces/deid_demo", metadata["app_file"]).is_file()
    assert metadata["sdk_version"] == "6.20.0"


def test_deid_demo_requirements_are_pinned():
    requirements = Path("examples/spaces/deid_demo/requirements.txt").read_text(
        encoding="utf-8"
    )
    lines = [
        line for line in requirements.splitlines() if line and not line.startswith("#")
    ]

    assert lines
    assert all(
        re.fullmatch(r"[A-Za-z0-9_.-]+(?:\[[A-Za-z0-9_,.-]+\])?==[^\s]+", line)
        for line in lines
    )


def test_deid_demo_readme_has_local_run_instructions():
    readme = Path("examples/spaces/deid_demo/README.md").read_text(encoding="utf-8")

    assert "python -m pip install -r requirements.txt" in readme
    assert "python app.py" in readme
