"""Offline-safe environment diagnostics for OpenMed."""

from __future__ import annotations

import importlib
import os
import platform
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .manifest_schema import MANIFEST_PATH
from .offline import env_flag_enabled

# Optional extras mapping with their actual import names
OPTIONAL_EXTRAS = {
    "mlx": "mlx",
    "coreml": "coremltools",
    "onnx": "onnxruntime",
    "hf": "transformers",
    "multimodal": "PIL",
}

# Known architecture mappings for validation
SUPPORTED_ARCHS = frozenset(
    {
        "x86_64",
        "AMD64",  # Intel/AMD 64-bit
        "arm64",
        "aarch64",  # ARM 64-bit (Apple Silicon, ARM servers)
        "armv7l",  # 32-bit ARM
    }
)

MIN_PYTHON_VERSION = (3, 10)
DEFAULT_HF_ENDPOINT = "https://huggingface.co"
PROXY_ENV_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY")


def _check(
    name: str,
    status: str,
    details: str,
    hint: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": name,
        "status": status,
        "details": details,
    }
    if hint is not None:
        payload["hint"] = hint
    return payload


def run_diagnostics() -> list[dict[str, Any]]:
    """Run local OpenMed diagnostics without network calls or secret exposure.

    Returns:
        A list of check dictionaries with ``name``, ``status``, ``details``,
        and optional remediation ``hint`` fields.
    """
    checks: list[dict[str, Any]] = []

    python_version = platform.python_version()
    python_supported = sys.version_info[:2] >= MIN_PYTHON_VERSION
    checks.append(
        _check(
            "python_version",
            "PASS" if python_supported else "FAIL",
            python_version,
            None if python_supported else "Use Python 3.10 or newer.",
        )
    )

    arch = platform.machine() or "unknown"
    arch_supported = arch in SUPPORTED_ARCHS
    checks.append(
        _check(
            "python_arch",
            "PASS" if arch_supported else "FAIL",
            arch,
            None if arch_supported else "Use a supported 64-bit Python architecture.",
        )
    )

    _check_openmed_version(checks)
    _check_optional_dependencies(checks)
    _check_hf_token(checks)
    _check_network_environment(checks)
    _check_offline_mode(checks)
    _check_manifest(checks)

    return checks


def _check_openmed_version(checks: list[dict[str, Any]]) -> None:
    try:
        from ..__about__ import __version__

        checks.append(_check("openmed_version", "PASS", __version__))
    except Exception:
        checks.append(_check("openmed_version", "WARN", "version unavailable"))


def _check_optional_dependencies(checks: list[dict[str, Any]]) -> None:
    hint_map = {
        "mlx": "Install with: pip install mlx",
        "coreml": "Install with: pip install coremltools",
        "onnx": "Install with: pip install onnxruntime",
        "hf": "Install with: pip install transformers",
        "multimodal": "Install with: pip install pillow",
    }

    for name, module_name in OPTIONAL_EXTRAS.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            checks.append(
                _check(
                    name,
                    "WARN",
                    f"{module_name} not installed",
                    hint_map.get(name),
                )
            )
            continue

        details = "Pillow installed" if name == "multimodal" else "installed"
        checks.append(_check(name, "PASS", details))


def _check_hf_token(checks: list[dict[str, Any]]) -> None:
    token_present = bool(os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN"))
    check = _check(
        "hf_token",
        "PASS" if token_present else "WARN",
        f"present={token_present}",
        None
        if token_present
        else "Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN environment variable",
    )
    check["present"] = token_present
    checks.append(check)


def _check_network_environment(checks: list[dict[str, Any]]) -> None:
    endpoint = os.getenv("HF_ENDPOINT") or DEFAULT_HF_ENDPOINT
    endpoint_check = _check("hf_endpoint", "PASS", _redact_url_credentials(endpoint))
    endpoint_check["source"] = "HF_ENDPOINT" if os.getenv("HF_ENDPOINT") else "default"
    checks.append(endpoint_check)

    for env_name in PROXY_ENV_VARS:
        value, source = _environment_value(env_name)
        details = "not set" if value is None else _redact_url_credentials(value)
        proxy_check = _check(env_name.lower(), "PASS", details)
        proxy_check["present"] = value is not None
        if source is not None:
            proxy_check["source"] = source
        checks.append(proxy_check)

    cache_path, cache_source = _hf_cache_location()
    cache_check = _check("hf_cache", "PASS", str(cache_path))
    cache_check["source"] = cache_source
    checks.append(cache_check)


def _environment_value(name: str) -> tuple[str | None, str | None]:
    for candidate in (name, name.lower()):
        value = os.getenv(candidate)
        if value:
            return value, candidate
    return None, None


def _redact_url_credentials(value: str) -> str:
    """Redact URL userinfo while preserving a useful diagnostic address."""
    try:
        parsed = urlsplit(value)
    except ValueError:
        return "configured (value could not be parsed safely)"
    if parsed.username is None:
        return value

    address = parsed.netloc.rsplit("@", 1)[-1]
    redacted_userinfo = "***:***" if parsed.password is not None else "***"
    return urlunsplit(parsed._replace(netloc=f"{redacted_userinfo}@{address}"))


def _hf_cache_location() -> tuple[Path, str]:
    explicit_cache = os.getenv("HF_HUB_CACHE")
    if explicit_cache:
        return Path(explicit_cache).expanduser(), "HF_HUB_CACHE"

    hf_home = os.getenv("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub", "HF_HOME"

    xdg_cache = os.getenv("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache).expanduser() / "huggingface" / "hub", "XDG_CACHE_HOME"

    return Path.home() / ".cache" / "huggingface" / "hub", "default"


def _check_offline_mode(checks: list[dict[str, Any]]) -> None:
    raw_value = os.getenv("OPENMED_OFFLINE")
    enabled = env_flag_enabled(raw_value)
    details = (
        f"{'enabled' if enabled else 'disabled'} (OPENMED_OFFLINE={raw_value or '0'})"
    )
    check = _check("openmed_offline", "PASS", details)
    check["enabled"] = enabled
    checks.append(check)


def _check_manifest(checks: list[dict[str, Any]]) -> None:
    if not MANIFEST_PATH.exists():
        checks.append(
            _check(
                "manifest_exists",
                "WARN",
                f"{MANIFEST_PATH.name} not found",
            )
        )
        checks.append(
            _check(
                "manifest_rows",
                "WARN",
                "not checked because manifest is missing",
            )
        )
        return

    checks.append(_check("manifest_exists", "PASS", str(MANIFEST_PATH)))

    try:
        with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
            rows = sum(1 for line in handle if line.strip())
    except OSError as exc:
        checks.append(_check("manifest_rows", "WARN", f"error reading manifest: {exc}"))
        return

    checks.append(_check("manifest_rows", "PASS", f"{rows} rows"))
