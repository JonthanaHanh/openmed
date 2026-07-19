"""Optional MuRIL and IndicBERT backbone-only encoder integration.

Weights are never bundled with OpenMed. A caller must provide a local path or
repository identifier before this module imports the optional runtime stack or
attempts to resolve any model files.
"""

from __future__ import annotations

import hashlib
import importlib
import logging
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from openmed.core.model_registry import (
    INDIC_ENCODER_SPECS,
    IndicEncoderLoadResult,
    IndicEncoderSpec,
    get_indic_encoder_spec,
)

from .base import EncoderOutput

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "Install the Hugging Face extra and a PyTorch runtime, for example "
    "`pip install 'openmed[hf]' torch`."
)


@dataclass
class IndicEncoderHandle:
    """Loaded tokenizer/backbone pair implementing the encoder contract."""

    source: str
    metadata: IndicEncoderSpec
    tokenizer: Any
    model: Any
    torch_module: Any
    device: str | None = None

    def encode(self, text: str, *, max_length: int = 512) -> EncoderOutput:
        """Encode one string without logging or returning the raw input text."""

        if not isinstance(text, str):
            raise TypeError("encoder input must be a string")
        if max_length <= 0:
            raise ValueError("max_length must be positive")

        encoded = self.tokenizer(
            text,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        tokenizer_outputs = dict(encoded)
        raw_offsets = tokenizer_outputs.pop("offset_mapping", None)
        if raw_offsets is None:
            raise ValueError(
                "configured tokenizer does not expose character offset mappings"
            )
        offsets = _normalize_offsets(raw_offsets)
        model_inputs = _move_to_device(tokenizer_outputs, self.device)

        inference_mode = getattr(self.torch_module, "inference_mode", None)
        context = inference_mode() if callable(inference_mode) else nullcontext()
        with context:
            model_output = self.model(**model_inputs, return_dict=True)

        hidden_states = _last_hidden_state(model_output)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        output = EncoderOutput(
            tokenizer_outputs=model_inputs,
            offset_mapping=offsets,
            last_hidden_state=hidden_states,
            text_sha256=digest,
        )
        output.validate()
        logger.debug(
            "Encoded Indic input sha256=%s characters=%d tokens=%d",
            digest,
            len(text),
            output.token_count,
        )
        return output


def is_indic_encoder_available() -> bool:
    """Return whether the optional Transformers and PyTorch stack imports."""

    return (
        _optional_module("transformers") is not None
        and _optional_module("torch") is not None
    )


def load_indic_encoder(
    source: str | None,
    *,
    family: str | None = None,
    cache_dir: str | Path | None = None,
    token: str | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
    device: str | None = None,
) -> IndicEncoderLoadResult:
    """Resolve user-supplied MuRIL or IndicBERT weights.

    A missing source, dependency, or checkpoint returns a deterministic skip
    reason. Repository identifiers may download only after the caller supplies
    one explicitly; existing filesystem paths are always loaded locally.
    """

    if source is None or not str(source).strip():
        return IndicEncoderLoadResult(
            handle=None,
            skip_reason="no Indic encoder weights are configured",
        )

    source_value = str(source).strip()
    metadata = get_indic_encoder_spec(family, source=source_value)
    transformers_module = _optional_module("transformers")
    torch_module = _optional_module("torch")
    if transformers_module is None or torch_module is None:
        missing = "transformers" if transformers_module is None else "torch"
        return IndicEncoderLoadResult(
            handle=None,
            skip_reason=f"optional dependency {missing!r} is unavailable; {_INSTALL_HINT}",
            metadata=metadata,
        )

    tokenizer_loader = getattr(transformers_module, "AutoTokenizer", None)
    model_loader = getattr(transformers_module, "AutoModel", None)
    if tokenizer_loader is None or model_loader is None:
        return IndicEncoderLoadResult(
            handle=None,
            skip_reason="the installed transformers package lacks AutoTokenizer/AutoModel",
            metadata=metadata,
        )

    load_kwargs: dict[str, Any] = {
        "local_files_only": bool(local_files_only or _is_existing_path(source_value)),
        "trust_remote_code": False,
    }
    if cache_dir is not None:
        load_kwargs["cache_dir"] = str(cache_dir)
    if token:
        load_kwargs["token"] = token
    if revision:
        load_kwargs["revision"] = revision

    try:
        tokenizer = tokenizer_loader.from_pretrained(
            source_value,
            use_fast=True,
            **load_kwargs,
        )
        model = model_loader.from_pretrained(source_value, **load_kwargs)
        if hasattr(model, "eval"):
            model.eval()
        if device and hasattr(model, "to"):
            model = model.to(device)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        return IndicEncoderLoadResult(
            handle=None,
            skip_reason=(
                f"configured {metadata.display_name} weights could not be loaded "
                f"({type(exc).__name__})"
            ),
            metadata=metadata,
        )

    return IndicEncoderLoadResult(
        handle=IndicEncoderHandle(
            source=source_value,
            metadata=metadata,
            tokenizer=tokenizer,
            model=model,
            torch_module=torch_module,
            device=device,
        ),
        skip_reason=None,
        metadata=metadata,
    )


def _optional_module(name: str) -> Any | None:
    try:
        return importlib.import_module(name)
    except (ImportError, OSError):
        return None


def _is_existing_path(source: str) -> bool:
    try:
        return Path(source).expanduser().exists()
    except OSError:
        return False


def _normalize_offsets(value: Any) -> tuple[tuple[int, int], ...]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if (
        isinstance(value, (list, tuple))
        and len(value) == 1
        and isinstance(value[0], (list, tuple))
    ):
        value = value[0]
    if not isinstance(value, (list, tuple)):
        raise ValueError("encoder offsets must be a single batched sequence")
    try:
        return tuple((int(offset[0]), int(offset[1])) for offset in value)
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError("encoder offsets contain an invalid span") from exc


def _move_to_device(
    tokenizer_outputs: Mapping[str, Any],
    device: str | None,
) -> dict[str, Any]:
    if not device:
        return dict(tokenizer_outputs)
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in tokenizer_outputs.items()
    }


def _last_hidden_state(model_output: Any) -> Any:
    hidden_states = getattr(model_output, "last_hidden_state", None)
    if hidden_states is not None:
        return hidden_states
    if isinstance(model_output, Mapping) and "last_hidden_state" in model_output:
        return model_output["last_hidden_state"]
    if isinstance(model_output, (list, tuple)) and model_output:
        return model_output[0]
    raise ValueError("configured encoder did not return last_hidden_state")


__all__ = [
    "INDIC_ENCODER_SPECS",
    "IndicEncoderHandle",
    "IndicEncoderLoadResult",
    "IndicEncoderSpec",
    "get_indic_encoder_spec",
    "is_indic_encoder_available",
    "load_indic_encoder",
]
