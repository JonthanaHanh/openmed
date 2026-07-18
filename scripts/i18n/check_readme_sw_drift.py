#!/usr/bin/env python3
"""Detect README.md changes not reviewed in the Swahili translation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = Path("docs/i18n/readme_sw_section_hashes.json")
SOURCE_README = Path("README.md")
TRANSLATION_README = Path("README.sw.md")
PREAMBLE = "__preamble__"

_H2 = re.compile(r"^##\s+(.+?)\s*$")
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
_MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+[^)]*)?\)")
_HTML_LINK = re.compile(r"(?:href|src)=[\"']([^\"']+)[\"']")


class DriftError(RuntimeError):
    """Raised when the Swahili README or its manifest is stale."""


@dataclass(frozen=True)
class Section:
    """One README preamble or H2 section."""

    heading: str
    content: str


def split_h2_sections(text: str) -> list[Section]:
    """Split Markdown into a preamble and fenced-code-aware H2 sections."""
    sections: list[Section] = []
    heading = PREAMBLE
    lines: list[str] = []
    fence: str | None = None

    for line in text.splitlines(keepends=True):
        fence_match = _FENCE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            if fence is None:
                fence = marker[0]
            elif marker[0] == fence:
                fence = None

        heading_match = _H2.match(line) if fence is None else None
        if heading_match:
            sections.append(Section(heading=heading, content="".join(lines)))
            heading = heading_match.group(1)
            lines = []
        else:
            lines.append(line)

    sections.append(Section(heading=heading, content="".join(lines)))
    return sections


def section_sha256(section: Section) -> str:
    """Return a stable SHA-256 digest for a section body."""
    normalized = section.content.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _read_sections(path: Path) -> list[Section]:
    if not path.is_file():
        raise DriftError(f"README file does not exist: {path}")
    return split_h2_sections(path.read_text(encoding="utf-8"))


def _relative_targets(text: str) -> set[str]:
    targets = {match.group(1) for match in _MARKDOWN_LINK.finditer(text)}
    targets.update(match.group(1) for match in _HTML_LINK.finditer(text))
    return targets


def validate_relative_links(root: Path, readme: Path) -> None:
    """Raise when a local link or image in ``readme`` does not resolve."""
    missing: list[str] = []
    text = readme.read_text(encoding="utf-8")
    for target in sorted(_relative_targets(text)):
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or target.startswith(("#", "/", "//")):
            continue
        relative_path = unquote(parsed.path)
        if relative_path and not (root / relative_path).exists():
            missing.append(target)

    if missing:
        formatted = "\n".join(f"- {readme.name}: {target}" for target in missing)
        raise DriftError(f"README contains unresolved local links:\n{formatted}")


def _validate_structure(
    source_path: Path,
    source_sections: list[Section],
    translation_path: Path,
    translation_sections: list[Section],
) -> None:
    if len(source_sections) != len(translation_sections):
        raise DriftError(
            f"{translation_path.name} has {len(translation_sections) - 1} H2 "
            f"sections; {source_path.name} has {len(source_sections) - 1}. "
            "Add a Swahili counterpart for every source H2 section."
        )

    if f'href="{translation_path.name}"' not in source_sections[0].content:
        raise DriftError(
            f"{source_path.name} language switcher must link to "
            f"{translation_path.name}."
        )
    if f'href="{source_path.name}"' not in translation_sections[0].content:
        raise DriftError(
            f"{translation_path.name} language switcher must link to "
            f"{source_path.name}."
        )


def build_manifest(root: Path) -> dict[str, object]:
    """Build a manifest after the source and translation were reviewed."""
    source_path = root / SOURCE_README
    translation_path = root / TRANSLATION_README
    source_sections = _read_sections(source_path)
    translation_sections = _read_sections(translation_path)
    _validate_structure(
        source_path,
        source_sections,
        translation_path,
        translation_sections,
    )
    validate_relative_links(root, translation_path)

    entries = []
    for source, translation in zip(source_sections, translation_sections, strict=True):
        entries.append(
            {
                "source_heading": source.heading,
                "translation_heading": translation.heading,
                "source_sha256": section_sha256(source),
                "translation_sha256": section_sha256(translation),
            }
        )

    return {
        "version": 1,
        "source": SOURCE_README.as_posix(),
        "translation": TRANSLATION_README.as_posix(),
        "sections": entries,
    }


def _load_manifest(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise DriftError(
            f"Swahili README manifest is missing: {path}. Review "
            "README.sw.md, then run check_readme_sw_drift.py --update."
        )
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise DriftError(f"Cannot read Swahili README manifest: {exc}") from exc
    if manifest.get("version") != 1:
        raise DriftError("Swahili README manifest must use version 1.")
    return manifest


def check_repository(root: Path, manifest_path: Path) -> None:
    """Validate H2 parity, reviewed hashes, switchers, and local links."""
    manifest = _load_manifest(manifest_path)
    if manifest.get("source") != SOURCE_README.as_posix():
        raise DriftError(f"Manifest source must be {SOURCE_README}.")
    if manifest.get("translation") != TRANSLATION_README.as_posix():
        raise DriftError(f"Manifest translation must be {TRANSLATION_README}.")

    source_path = root / SOURCE_README
    translation_path = root / TRANSLATION_README
    source_sections = _read_sections(source_path)
    translation_sections = _read_sections(translation_path)
    _validate_structure(
        source_path,
        source_sections,
        translation_path,
        translation_sections,
    )
    validate_relative_links(root, translation_path)

    entries = manifest.get("sections")
    if not isinstance(entries, list) or len(entries) != len(source_sections):
        raise DriftError("Swahili README manifest has the wrong section count.")

    errors: list[str] = []
    for source, translation, entry in zip(
        source_sections, translation_sections, entries, strict=True
    ):
        if not isinstance(entry, dict):
            errors.append("Manifest contains an invalid section entry.")
            continue
        if entry.get("source_heading") != source.heading:
            errors.append(
                f"README.sw.md has no reviewed entry for source heading "
                f"{source.heading!r}."
            )
            continue
        if entry.get("translation_heading") != translation.heading:
            errors.append(
                f"README.sw.md heading changed for source heading {source.heading!r}."
            )
        if entry.get("source_sha256") != section_sha256(source):
            errors.append(
                f"README.sw.md is stale for source heading {source.heading!r}; "
                "README.md changed."
            )
        if entry.get("translation_sha256") != section_sha256(translation):
            errors.append(
                f"README.sw.md manifest is stale for heading {translation.heading!r}."
            )

    if errors:
        details = "\n".join(f"- {error}" for error in errors)
        raise DriftError(
            "Swahili README drift detected:\n"
            f"{details}\n"
            "Review the paired sections, then run "
            "`python scripts/i18n/check_readme_sw_drift.py --update`."
        )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repository root (defaults to the root containing this script).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Manifest path, relative to --root unless absolute.",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Refresh hashes after reviewing README.sw.md against README.md.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the drift check or refresh its reviewed section manifest."""
    args = _parse_args(argv)
    root = args.root.resolve()
    manifest_path = args.manifest
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path

    try:
        if args.update:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(build_manifest(root), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"Updated {manifest_path.relative_to(root)}")
        check_repository(root, manifest_path)
    except DriftError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("Swahili README drift check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
