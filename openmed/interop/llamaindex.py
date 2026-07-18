"""LlamaIndex redaction and tool adapters backed by OpenMed."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from importlib import import_module as _import_module
from pathlib import Path
from typing import Any

from openmed.interop.function_tools import (
    RuntimeProvider,
    create_tool_callable,
    registry_tool_specs,
)
from openmed.mcp.tool_registry import render_adapter_tool_definitions

Deidentifier = Callable[..., Any]


@dataclass(frozen=True)
class LlamaIndexRedactionConfig:
    """Runtime options forwarded to OpenMed's de-identification engine."""

    method: str = "mask"
    model_name: str | None = None
    confidence_threshold: float = 0.7
    keep_year: bool = False
    keep_mapping: bool = False
    use_smart_merging: bool = True
    lang: str = "en"
    normalize_accents: bool | None = None
    use_safety_sweep: bool = True
    consistent: bool = False
    seed: int | None = None
    locale: str | None = None
    policy: str | None = None
    calibration_thresholds_path: str | Path | None = None
    extra_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def to_deidentify_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments for ``openmed.core.pii.deidentify``."""

        kwargs: dict[str, Any] = {
            "method": self.method,
            "confidence_threshold": self.confidence_threshold,
            "keep_year": self.keep_year,
            "keep_mapping": self.keep_mapping,
            "use_smart_merging": self.use_smart_merging,
            "lang": self.lang,
            "normalize_accents": self.normalize_accents,
            "use_safety_sweep": self.use_safety_sweep,
            "consistent": self.consistent,
            "seed": self.seed,
            "locale": self.locale,
            "policy": self.policy,
            "calibration_thresholds_path": self.calibration_thresholds_path,
        }
        if self.model_name is not None:
            kwargs["model_name"] = self.model_name

        kwargs.update(dict(self.extra_kwargs))
        return {key: value for key, value in kwargs.items() if value is not None}


class _NodeRedactor:
    def __init__(
        self,
        *,
        config: LlamaIndexRedactionConfig,
        deidentifier: Deidentifier | None,
    ) -> None:
        self.config = config
        self._deidentifier = deidentifier

    def redact_scored_nodes(self, nodes: Sequence[Any]) -> list[Any]:
        return [self._redact_scored_node(node) for node in nodes]

    def redact_nodes(self, nodes: Sequence[Any]) -> list[Any]:
        return [self._redact_node(node) for node in nodes]

    def _redact_scored_node(self, scored_node: Any) -> Any:
        try:
            redacted = _clone(scored_node)
            self._redact_node(redacted.node, clone=False)
        except AttributeError as exc:
            raise TypeError(
                "LlamaIndex postprocessor inputs must expose a node attribute"
            ) from exc
        return redacted

    def _redact_node(self, node: Any, *, clone: bool = True) -> Any:
        redacted = _clone(node) if clone else node
        text = _node_text(redacted)
        if text is None or text == "":
            return redacted

        result = self._deidentifier_or_default()(
            text,
            **self.config.to_deidentify_kwargs(),
        )
        redacted_text = _deidentified_text(result)
        _set_node_text(redacted, redacted_text)
        return redacted

    def _deidentifier_or_default(self) -> Deidentifier:
        if self._deidentifier is not None:
            return self._deidentifier

        from openmed.core.pii import deidentify

        return deidentify


def create_redaction_postprocessor(
    *,
    config: LlamaIndexRedactionConfig | None = None,
    deidentifier: Deidentifier | None = None,
) -> Any:
    """Create a LlamaIndex node postprocessor that redacts retrieved text."""

    base = _load_optional_class(
        "llama_index.core.postprocessor.types",
        "BaseNodePostprocessor",
        feature="node postprocessors",
    )
    redactor = _NodeRedactor(
        config=config or LlamaIndexRedactionConfig(),
        deidentifier=deidentifier,
    )

    class OpenMedRedactionPostprocessor(base):
        @classmethod
        def class_name(cls) -> str:
            return "OpenMedRedactionPostprocessor"

        def __init__(self) -> None:
            super().__init__()
            object.__setattr__(self, "_openmed_redactor", redactor)

        def _postprocess_nodes(
            self,
            nodes: list[Any],
            query_bundle: Any | None = None,
        ) -> list[Any]:
            del query_bundle
            return self._openmed_redactor.redact_scored_nodes(nodes)

    OpenMedRedactionPostprocessor.__module__ = __name__
    return OpenMedRedactionPostprocessor()


def create_redaction_transform(
    *,
    config: LlamaIndexRedactionConfig | None = None,
    deidentifier: Deidentifier | None = None,
) -> Any:
    """Create an optional ingestion transform that redacts nodes before storage."""

    base = _load_optional_class(
        "llama_index.core.schema",
        "TransformComponent",
        feature="ingestion transforms",
    )
    redactor = _NodeRedactor(
        config=config or LlamaIndexRedactionConfig(),
        deidentifier=deidentifier,
    )

    class OpenMedRedactionTransform(base):
        @classmethod
        def class_name(cls) -> str:
            return "OpenMedRedactionTransform"

        def __init__(self) -> None:
            super().__init__()
            object.__setattr__(self, "_openmed_redactor", redactor)

        def __call__(self, nodes: Sequence[Any], **kwargs: Any) -> list[Any]:
            del kwargs
            return self._openmed_redactor.redact_nodes(nodes)

    OpenMedRedactionTransform.__module__ = __name__
    return OpenMedRedactionTransform()


def create_tool_definitions() -> tuple[dict[str, Any], ...]:
    """Return LlamaIndex-facing OpenMed tool definitions from the registry."""

    return render_adapter_tool_definitions("llamaindex")


def get_llamaindex_tools(
    *,
    runtime_provider: RuntimeProvider | None = None,
) -> tuple[Any, ...]:
    """Return LlamaIndex ``FunctionTool`` objects for every registry tool."""

    function_tool = _load_function_tool()
    return tuple(
        _function_tool_from_spec(function_tool, spec, runtime_provider)
        for spec in registry_tool_specs()
    )


def _function_tool_from_spec(
    function_tool: Any,
    spec: Any,
    runtime_provider: RuntimeProvider | None,
) -> Any:
    func = create_tool_callable(spec, runtime_provider=runtime_provider)
    if hasattr(function_tool, "from_defaults"):
        return function_tool.from_defaults(
            fn=func,
            name=spec.name,
            description=spec.description,
        )
    if hasattr(function_tool, "from_function"):
        return function_tool.from_function(
            func=func,
            name=spec.name,
            description=spec.description,
        )
    raise ImportError("LlamaIndex tools require llama-index-core with FunctionTool.")


def _load_function_tool() -> Any:
    return _load_optional_class(
        "llama_index.core.tools",
        "FunctionTool",
        feature="tools",
    )


def _load_optional_class(module_name: str, class_name: str, *, feature: str) -> Any:
    try:
        module = _import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"LlamaIndex {feature} require the 'llamaindex' extra. "
            "Install with `pip install openmed[llamaindex]`."
        ) from exc

    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(
            f"LlamaIndex {feature} require llama-index-core with {class_name}."
        ) from exc


def _clone(value: Any) -> Any:
    model_copy = getattr(value, "model_copy", None)
    if callable(model_copy):
        return model_copy(deep=True)
    return deepcopy(value)


def _node_text(node: Any) -> str | None:
    text = getattr(node, "text", None)
    if isinstance(text, str):
        return text

    get_content = getattr(node, "get_content", None)
    if callable(get_content):
        content = get_content()
        if isinstance(content, str):
            return content
    return None


def _set_node_text(node: Any, text: str) -> None:
    set_content = getattr(node, "set_content", None)
    if callable(set_content):
        set_content(text)
        return
    if hasattr(node, "text"):
        node.text = text
        return
    raise TypeError("LlamaIndex node does not expose mutable text content")


def _deidentified_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    try:
        return str(result.deidentified_text)
    except AttributeError as exc:
        raise TypeError(
            "deidentifier must return a string or an object with deidentified_text"
        ) from exc


__all__ = [
    "Deidentifier",
    "LlamaIndexRedactionConfig",
    "create_redaction_postprocessor",
    "create_redaction_transform",
    "create_tool_definitions",
    "get_llamaindex_tools",
]
