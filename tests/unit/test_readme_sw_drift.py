"""Tests for the Swahili README translation drift guard."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.i18n.check_readme_sw_drift import (
    PREAMBLE,
    DriftError,
    build_manifest,
    check_repository,
    split_h2_sections,
    validate_relative_links,
)

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "docs" / "i18n" / "readme_sw_section_hashes.json"


def _write_manifest(root: Path, manifest_path: Path) -> None:
    manifest_path.write_text(
        json.dumps(build_manifest(root), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_fixture_repo(root: Path) -> Path:
    (root / "docs" / "i18n").mkdir(parents=True)
    (root / "docs" / "guide.md").write_text("# Guide\n", encoding="utf-8")
    (root / "README.md").write_text(
        '<a href="README.sw.md">Kiswahili</a>\n'
        "\n## First\n\nEnglish body.\n"
        "\n## Second\n\n[Guide](docs/guide.md)\n",
        encoding="utf-8",
    )
    (root / "README.sw.md").write_text(
        '<a href="README.md">English</a>\n'
        "\n## Kwanza\n\nMaudhui ya Kiswahili.\n"
        "\n## Pili\n\n[Mwongozo](docs/guide.md)\n",
        encoding="utf-8",
    )
    manifest_path = root / "docs" / "i18n" / "readme_sw_section_hashes.json"
    _write_manifest(root, manifest_path)
    return manifest_path


def test_split_h2_sections_ignores_headings_inside_fences() -> None:
    sections = split_h2_sections(
        "intro\n\n## Real\n\n```markdown\n## Not a section\n```\n"
    )

    assert [section.heading for section in sections] == [PREAMBLE, "Real"]
    assert "## Not a section" in sections[1].content


def test_source_heading_addition_fails_until_swahili_is_updated(
    tmp_path: Path,
) -> None:
    manifest_path = _write_fixture_repo(tmp_path)
    source = tmp_path / "README.md"
    source.write_text(
        source.read_text(encoding="utf-8") + "\n## Third\n\nNew source section.\n",
        encoding="utf-8",
    )

    with pytest.raises(DriftError, match="H2 sections"):
        check_repository(tmp_path, manifest_path)

    translation = tmp_path / "README.sw.md"
    translation.write_text(
        translation.read_text(encoding="utf-8")
        + "\n## Tatu\n\nSehemu mpya ya Kiswahili.\n",
        encoding="utf-8",
    )
    _write_manifest(tmp_path, manifest_path)

    check_repository(tmp_path, manifest_path)


def test_source_body_edit_requires_reviewed_manifest_refresh(tmp_path: Path) -> None:
    manifest_path = _write_fixture_repo(tmp_path)
    source = tmp_path / "README.md"
    source.write_text(
        source.read_text(encoding="utf-8").replace(
            "English body.", "Changed English body."
        ),
        encoding="utf-8",
    )

    with pytest.raises(DriftError, match="README.sw.md is stale"):
        check_repository(tmp_path, manifest_path)


def test_translation_edit_requires_manifest_refresh(tmp_path: Path) -> None:
    manifest_path = _write_fixture_repo(tmp_path)
    translation = tmp_path / "README.sw.md"
    translation.write_text(
        translation.read_text(encoding="utf-8").replace(
            "Maudhui ya Kiswahili.", "Maudhui yaliyosasishwa."
        ),
        encoding="utf-8",
    )

    with pytest.raises(DriftError, match="manifest is stale"):
        check_repository(tmp_path, manifest_path)


def test_relative_link_validation_reports_missing_target(tmp_path: Path) -> None:
    readme = tmp_path / "README.sw.md"
    readme.write_text("[Hati](docs/missing.md)\n", encoding="utf-8")

    with pytest.raises(DriftError, match="docs/missing.md"):
        validate_relative_links(tmp_path, readme)


def test_committed_readmes_and_manifest_are_in_sync() -> None:
    check_repository(ROOT, MANIFEST)

    source_sections = split_h2_sections(
        (ROOT / "README.md").read_text(encoding="utf-8")
    )
    translation_sections = split_h2_sections(
        (ROOT / "README.sw.md").read_text(encoding="utf-8")
    )
    assert len(source_sections) == len(translation_sections)


def test_ci_runs_swahili_drift_check_for_readme_changes() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "readme-sw-drift:" in workflow
    assert "README.md README.sw.md" in workflow
    assert "python scripts/i18n/check_readme_sw_drift.py" in workflow
