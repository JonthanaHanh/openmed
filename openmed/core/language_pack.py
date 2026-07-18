"""Minimal language-pack contract for script-aware surrogate dispatch.

The broader language integration surface can grow independently, but surrogate
providers need one stable process-local contract today: a language code, the
Unicode scripts it owns, and the conceptual locale used by its fallback
generator.  The registry deliberately stores declarations only; label-specific
generator functions remain in :mod:`openmed.core.anonymizer.registry` to avoid
an import cycle.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from threading import RLock

_LANGUAGE_CODE = re.compile(r"^[a-z]{2}$")


@dataclass(frozen=True, slots=True)
class LanguagePack:
    """Describe the language metadata needed for surrogate generation.

    Args:
        code: Lowercase ISO 639-1 language code.
        scripts: Unicode script names accepted by script-aware providers.
        surrogate_locale: Conceptual Faker locale used by fallback generators.
    """

    code: str
    scripts: tuple[str, ...]
    surrogate_locale: str

    def __post_init__(self) -> None:
        """Validate and freeze one language-pack declaration."""

        if not isinstance(self.code, str) or not _LANGUAGE_CODE.fullmatch(self.code):
            raise ValueError("code must be a lowercase ISO 639-1 language code")
        if isinstance(self.scripts, str):
            raise TypeError("scripts must be an iterable of script names")
        scripts = tuple(self.scripts)
        if not scripts or any(
            not isinstance(script, str) or not script.strip() for script in scripts
        ):
            raise ValueError("scripts must contain non-empty script names")
        if len(set(scripts)) != len(scripts):
            raise ValueError("scripts must not contain duplicates")
        if not isinstance(self.surrogate_locale, str) or not self.surrogate_locale:
            raise ValueError("surrogate_locale must be a non-empty string")
        object.__setattr__(self, "scripts", scripts)


class LanguagePackRegistry:
    """Store immutable language packs in a thread-safe process-local registry."""

    def __init__(self) -> None:
        """Create an empty registry."""

        self._packs: dict[str, LanguagePack] = {}
        self._lock = RLock()

    def register(
        self,
        pack: LanguagePack,
        *,
        replace: bool = False,
    ) -> LanguagePack:
        """Register and return ``pack``.

        Re-registering an identical immutable declaration is idempotent, which
        keeps module reloads and provider bundles safe. A conflicting
        declaration requires an explicit replacement.
        """

        if not isinstance(pack, LanguagePack):
            raise TypeError("pack must be a LanguagePack")
        with self._lock:
            current = self._packs.get(pack.code)
            if current is not None and current != pack and not replace:
                raise ValueError(f"language pack {pack.code!r} is already registered")
            self._packs[pack.code] = pack
        return pack

    def get(self, code: str) -> LanguagePack:
        """Return the pack registered for ``code``.

        Raises:
            KeyError: If ``code`` has no registered language pack.
        """

        with self._lock:
            return self._packs[code]

    def find(self, code: str) -> LanguagePack | None:
        """Return the pack registered for ``code``, if any."""

        with self._lock:
            return self._packs.get(code)

    def iter_codes(self) -> Iterator[str]:
        """Iterate over a stable, sorted snapshot of registered codes."""

        with self._lock:
            return iter(tuple(sorted(self._packs)))


LANGUAGE_PACK_REGISTRY = LanguagePackRegistry()


def register_language_pack(
    pack: LanguagePack,
    *,
    replace: bool = False,
) -> LanguagePack:
    """Register a process-local language pack."""

    return LANGUAGE_PACK_REGISTRY.register(pack, replace=replace)


def get_language_pack(code: str) -> LanguagePack | None:
    """Return a process-local language pack without raising for unknown codes."""

    return LANGUAGE_PACK_REGISTRY.find(code)


__all__ = [
    "LANGUAGE_PACK_REGISTRY",
    "LanguagePack",
    "LanguagePackRegistry",
    "get_language_pack",
    "register_language_pack",
]
