from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from openmed.interop import adapter_spec, get_adapter
from openmed.interop import llamaindex as llamaindex_adapter


@dataclass
class FixtureNode:
    text: str
    metadata: dict[str, str]

    def set_content(self, text: str) -> None:
        self.text = text


@dataclass
class FixtureNodeWithScore:
    node: FixtureNode
    score: float


class BaseNodePostprocessorLike:
    callback_manager = None

    def postprocess_nodes(self, nodes, query_bundle=None, query_str=None):
        del query_str
        return self._postprocess_nodes(nodes, query_bundle=query_bundle)

    async def apostprocess_nodes(self, nodes, query_bundle=None, query_str=None):
        del query_str
        return self._postprocess_nodes(nodes, query_bundle=query_bundle)


class TransformComponentLike:
    async def acall(self, nodes, **kwargs):
        return self(nodes, **kwargs)


def fake_import(name: str):
    if name == "llama_index.core.postprocessor.types":
        return SimpleNamespace(BaseNodePostprocessor=BaseNodePostprocessorLike)
    if name == "llama_index.core.schema":
        return SimpleNamespace(TransformComponent=TransformComponentLike)
    raise ImportError(name)


def fake_deidentify(text: str, **kwargs):
    assert kwargs["method"] == "mask"
    assert kwargs["keep_year"] is False
    assert kwargs["use_safety_sweep"] is True
    redacted = (
        text.replace("Jane Roe", "[PERSON]")
        .replace("jane.roe@example.com", "[EMAIL]")
        .replace("555-0100", "[PHONE]")
    )
    return SimpleNamespace(deidentified_text=redacted)


def test_registry_loads_llamaindex_redaction_adapter_lazily() -> None:
    for name in list(sys.modules):
        if name == "llama_index" or name.startswith("llama_index."):
            sys.modules.pop(name, None)

    adapter = get_adapter("llamaindex")

    assert adapter is llamaindex_adapter
    assert adapter_spec("llamaindex").extra == "llamaindex"
    assert hasattr(adapter, "create_redaction_postprocessor")
    assert not any(
        name == "llama_index" or name.startswith("llama_index.") for name in sys.modules
    )


def test_postprocessor_redacts_fixture_nodes_without_an_llm(monkeypatch) -> None:
    monkeypatch.setattr(llamaindex_adapter, "_import_module", fake_import)
    original = FixtureNodeWithScore(
        node=FixtureNode(
            text="Patient Jane Roe called jane.roe@example.com or 555-0100.",
            metadata={"source": "synthetic-fixture"},
        ),
        score=0.93,
    )

    postprocessor = llamaindex_adapter.create_redaction_postprocessor(
        deidentifier=fake_deidentify,
    )
    processed = postprocessor.postprocess_nodes([original])

    assert isinstance(postprocessor, BaseNodePostprocessorLike)
    assert processed[0].node.text == ("Patient [PERSON] called [EMAIL] or [PHONE].")
    assert processed[0].node.metadata == {"source": "synthetic-fixture"}
    assert processed[0].score == pytest.approx(0.93)
    assert original.node.text == (
        "Patient Jane Roe called jane.roe@example.com or 555-0100."
    )


def test_postprocessor_supports_async_retrieval(monkeypatch) -> None:
    monkeypatch.setattr(llamaindex_adapter, "_import_module", fake_import)
    postprocessor = llamaindex_adapter.create_redaction_postprocessor(
        deidentifier=fake_deidentify,
    )
    original = FixtureNodeWithScore(
        node=FixtureNode("Jane Roe", {}),
        score=0.7,
    )

    processed = asyncio.run(postprocessor.apostprocess_nodes([original]))

    assert processed[0].node.text == "[PERSON]"


def test_ingestion_transform_redacts_copies_before_storage(monkeypatch) -> None:
    monkeypatch.setattr(llamaindex_adapter, "_import_module", fake_import)
    original = FixtureNode(
        text="Jane Roe can be reached at 555-0100.",
        metadata={"source": "synthetic-fixture"},
    )

    transform = llamaindex_adapter.create_redaction_transform(
        deidentifier=fake_deidentify,
    )
    processed = transform([original])

    assert isinstance(transform, TransformComponentLike)
    assert processed[0].text == "[PERSON] can be reached at [PHONE]."
    assert processed[0].metadata == {"source": "synthetic-fixture"}
    assert original.text == "Jane Roe can be reached at 555-0100."


def test_redaction_factories_raise_clear_error_without_extra(monkeypatch) -> None:
    def missing_dependency(name: str):
        raise ImportError(name)

    monkeypatch.setattr(llamaindex_adapter, "_import_module", missing_dependency)

    with pytest.raises(ImportError, match=r"openmed\[llamaindex\]"):
        llamaindex_adapter.create_redaction_postprocessor(
            deidentifier=fake_deidentify,
        )
    with pytest.raises(ImportError, match=r"openmed\[llamaindex\]"):
        llamaindex_adapter.create_redaction_transform(deidentifier=fake_deidentify)
