"""Synthetic evaluation for token-LID-driven Hinglish PII routing.

The documented token-accuracy floor is 0.80, matching the issue's baseline
target. The suite also compares deterministic code-mixed pattern routing with
a single-language English baseline and requires zero residual gold entities.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openmed.core.lang_id_codemix import (
    TokenLanguage,
    TokenLIDHook,
    identify_token_languages,
)
from openmed.core.pii_entity_merger import PIIPattern, find_semantic_units
from openmed.core.pii_i18n import (
    get_code_mixed_pattern_runs,
    get_patterns_for_code_mixed_text,
    get_patterns_for_language,
)

CODE_MIXED_ROUTING = "code_mixed_routing"
MIN_TOKEN_LID_ACCURACY = 0.80
CODE_MIXED_ROUTING_FIXTURE_PATH = (
    Path(__file__).parents[1] / "golden" / "fixtures" / "code_mixed_hinglish.jsonl"
)

_GOLD_TO_PATTERN_LABEL = {
    "DATE": "date",
    "ID_NUM": "national_id",
    "PHONE": "phone_number",
    "ZIPCODE": "postcode",
}
_RAW_TOKEN_FIELDS = frozenset({"surface", "text", "token", "value"})


@dataclass(frozen=True)
class GoldTokenLanguage:
    """Offset-only token LID expectation."""

    start: int
    end: int
    label: str


@dataclass(frozen=True)
class CodeMixedGoldSpan:
    """Offset-only deterministic PII expectation."""

    start: int
    end: int
    label: str


@dataclass(frozen=True)
class CodeMixedRoutingFixture:
    """One synthetic Hinglish note and its offset-only gold annotations."""

    fixture_id: str
    text: str
    gold_tokens: tuple[GoldTokenLanguage, ...]
    gold_spans: tuple[CodeMixedGoldSpan, ...]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class CodeMixedRoutingResult:
    """Aggregate token LID and downstream routing metrics."""

    fixture_count: int
    token_count: int
    token_lid_accuracy: float
    minimum_token_lid_accuracy: float
    baseline_deid_recall: float
    code_mixed_deid_recall: float
    recall_improvement: float
    entity_leakage_count: int
    named_entity_token_count: int
    named_entity_tokens_retained: int
    deterministic: bool
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-ready metrics without fixture or token surfaces."""
        return {
            "baseline_deid_recall": self.baseline_deid_recall,
            "code_mixed_deid_recall": self.code_mixed_deid_recall,
            "deterministic": self.deterministic,
            "entity_leakage_count": self.entity_leakage_count,
            "fixture_count": self.fixture_count,
            "minimum_token_lid_accuracy": self.minimum_token_lid_accuracy,
            "named_entity_token_count": self.named_entity_token_count,
            "named_entity_tokens_retained": self.named_entity_tokens_retained,
            "passed": self.passed,
            "recall_improvement": self.recall_improvement,
            "token_count": self.token_count,
            "token_lid_accuracy": self.token_lid_accuracy,
        }


def load_code_mixed_routing_fixtures(
    path: str | Path | None = None,
) -> list[CodeMixedRoutingFixture]:
    """Load and validate the committed synthetic Hinglish fixture set."""
    fixture_path = Path(path) if path is not None else CODE_MIXED_ROUTING_FIXTURE_PATH
    fixtures: list[CodeMixedRoutingFixture] = []
    with fixture_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"fixture line {line_number} must be an object")
            fixtures.append(_fixture_from_mapping(payload, line_number=line_number))
    if not fixtures:
        raise ValueError("code-mixed routing fixture set is empty")
    fixture_ids = [fixture.fixture_id for fixture in fixtures]
    if len(fixture_ids) != len(set(fixture_ids)):
        raise ValueError("code-mixed routing fixture IDs must be unique")
    return fixtures


def code_mixed_routing_metadata(
    *,
    fixture_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return suite metadata documenting its accuracy and leakage gates."""
    return {
        "fixture_path": str(fixture_path or CODE_MIXED_ROUTING_FIXTURE_PATH),
        "minimum_token_lid_accuracy": MIN_TOKEN_LID_ACCURACY,
        "requires_measurable_recall_improvement": True,
        "requires_zero_entity_leakage": True,
        "suite": CODE_MIXED_ROUTING,
        "synthetic": True,
    }


def evaluate_code_mixed_routing(
    fixtures: Sequence[CodeMixedRoutingFixture],
    *,
    lid_model: TokenLIDHook | None = None,
) -> CodeMixedRoutingResult:
    """Score token LID, deterministic routing recall, and residual leakage."""
    token_total = 0
    token_correct = 0
    gold_span_total = 0
    baseline_hits = 0
    routed_hits = 0
    entity_leakage_count = 0
    named_entity_token_count = 0
    named_entity_tokens_retained = 0

    first_pass: list[tuple[TokenLanguage, ...]] = []
    second_pass: list[tuple[TokenLanguage, ...]] = []

    for fixture in fixtures:
        observed_tokens = identify_token_languages(fixture.text, model=lid_model)
        repeated_tokens = identify_token_languages(fixture.text, model=lid_model)
        first_pass.append(observed_tokens)
        second_pass.append(repeated_tokens)
        token_total += len(fixture.gold_tokens)
        token_correct += _correct_token_count(fixture.gold_tokens, observed_tokens)

        routes = get_code_mixed_pattern_runs(
            fixture.text,
            lid_model=lid_model,
        )
        for expected in fixture.gold_tokens:
            if expected.label != "ne":
                continue
            named_entity_token_count += 1
            if any(
                route.start <= expected.start
                and route.end >= expected.end
                and set(route.languages) == {"hi", "en"}
                and route.patterns
                for route in routes
            ):
                named_entity_tokens_retained += 1

        baseline = _observed_spans(
            fixture.text,
            get_patterns_for_language("en"),
        )
        routed = _observed_spans(
            fixture.text,
            get_patterns_for_code_mixed_text(
                fixture.text,
                lid_model=lid_model,
            ),
        )
        for gold_span in fixture.gold_spans:
            gold_span_total += 1
            expected = (
                gold_span.start,
                gold_span.end,
                _GOLD_TO_PATTERN_LABEL[gold_span.label],
            )
            if expected in baseline:
                baseline_hits += 1
            if expected in routed:
                routed_hits += 1
            else:
                entity_leakage_count += 1

    token_accuracy = _ratio(token_correct, token_total)
    baseline_recall = _ratio(baseline_hits, gold_span_total)
    routed_recall = _ratio(routed_hits, gold_span_total)
    recall_improvement = routed_recall - baseline_recall
    deterministic = first_pass == second_pass
    passed = (
        token_accuracy >= MIN_TOKEN_LID_ACCURACY
        and recall_improvement > 0.0
        and entity_leakage_count == 0
        and named_entity_tokens_retained == named_entity_token_count
        and deterministic
    )
    return CodeMixedRoutingResult(
        fixture_count=len(fixtures),
        token_count=token_total,
        token_lid_accuracy=token_accuracy,
        minimum_token_lid_accuracy=MIN_TOKEN_LID_ACCURACY,
        baseline_deid_recall=baseline_recall,
        code_mixed_deid_recall=routed_recall,
        recall_improvement=recall_improvement,
        entity_leakage_count=entity_leakage_count,
        named_entity_token_count=named_entity_token_count,
        named_entity_tokens_retained=named_entity_tokens_retained,
        deterministic=deterministic,
        passed=passed,
    )


def run_code_mixed_routing(
    *,
    fixture_path: str | Path | None = None,
    fixtures: Sequence[CodeMixedRoutingFixture] | None = None,
    lid_model: TokenLIDHook | None = None,
) -> CodeMixedRoutingResult:
    """Load the synthetic fixtures and run the code-mixed routing gates."""
    loaded = (
        tuple(fixtures)
        if fixtures is not None
        else tuple(load_code_mixed_routing_fixtures(fixture_path))
    )
    return evaluate_code_mixed_routing(loaded, lid_model=lid_model)


def _fixture_from_mapping(
    payload: Mapping[str, Any],
    *,
    line_number: int,
) -> CodeMixedRoutingFixture:
    fixture_id = str(payload.get("id", "")).strip()
    text = payload.get("text")
    metadata = payload.get("metadata")
    if not fixture_id or not isinstance(text, str) or not text:
        raise ValueError(f"fixture line {line_number} requires id and text")
    if not isinstance(metadata, Mapping) or metadata.get("synthetic") is not True:
        raise ValueError(f"fixture {fixture_id} must be marked synthetic")

    gold_tokens = tuple(
        _gold_token_from_mapping(item, fixture_id=fixture_id)
        for item in _mapping_sequence(payload.get("gold_tokens"), "gold_tokens")
    )
    gold_spans = tuple(
        _gold_span_from_mapping(item, fixture_id=fixture_id)
        for item in _mapping_sequence(payload.get("gold_spans"), "gold_spans")
    )
    fixture = CodeMixedRoutingFixture(
        fixture_id=fixture_id,
        text=text,
        gold_tokens=gold_tokens,
        gold_spans=gold_spans,
        metadata=dict(metadata),
    )
    _validate_fixture_offsets(fixture)
    return fixture


def _mapping_sequence(value: Any, field_name: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    if not all(isinstance(item, Mapping) for item in value):
        raise ValueError(f"{field_name} entries must be objects")
    return tuple(value)


def _gold_token_from_mapping(
    payload: Mapping[str, Any],
    *,
    fixture_id: str,
) -> GoldTokenLanguage:
    if _RAW_TOKEN_FIELDS & set(payload):
        raise ValueError(f"fixture {fixture_id} token gold must be offset-only")
    return GoldTokenLanguage(
        start=int(payload["start"]),
        end=int(payload["end"]),
        label=str(payload["label"]),
    )


def _gold_span_from_mapping(
    payload: Mapping[str, Any],
    *,
    fixture_id: str,
) -> CodeMixedGoldSpan:
    label = str(payload["label"])
    if label not in _GOLD_TO_PATTERN_LABEL:
        raise ValueError(f"fixture {fixture_id} has unsupported gold span label")
    return CodeMixedGoldSpan(
        start=int(payload["start"]),
        end=int(payload["end"]),
        label=label,
    )


def _validate_fixture_offsets(fixture: CodeMixedRoutingFixture) -> None:
    observed = identify_token_languages(fixture.text)
    expected_bounds = [(token.start, token.end) for token in fixture.gold_tokens]
    observed_bounds = [(token.start, token.end) for token in observed]
    if expected_bounds != observed_bounds:
        raise ValueError(f"fixture {fixture.fixture_id} token offsets do not align")
    for item in (*fixture.gold_tokens, *fixture.gold_spans):
        if not (0 <= item.start < item.end <= len(fixture.text)):
            raise ValueError(f"fixture {fixture.fixture_id} contains invalid offsets")


def _correct_token_count(
    expected: Sequence[GoldTokenLanguage],
    observed: Sequence[TokenLanguage],
) -> int:
    observed_map = {(token.start, token.end): token.label for token in observed}
    return sum(
        observed_map.get((token.start, token.end)) == token.label for token in expected
    )


def _observed_spans(
    text: str,
    patterns: Sequence[PIIPattern],
) -> set[tuple[int, int, str]]:
    observed: set[tuple[int, int, str]] = set()
    for unit in find_semantic_units(text, list(patterns)):
        start, end, label = int(unit[0]), int(unit[1]), str(unit[2])
        validated = bool(unit[5]) if len(unit) >= 6 else True
        if validated:
            observed.add((start, end, label))
    return observed


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 1.0


__all__ = [
    "CODE_MIXED_ROUTING",
    "CODE_MIXED_ROUTING_FIXTURE_PATH",
    "MIN_TOKEN_LID_ACCURACY",
    "CodeMixedGoldSpan",
    "CodeMixedRoutingFixture",
    "CodeMixedRoutingResult",
    "GoldTokenLanguage",
    "code_mixed_routing_metadata",
    "evaluate_code_mixed_routing",
    "load_code_mixed_routing_fixtures",
    "run_code_mixed_routing",
]
