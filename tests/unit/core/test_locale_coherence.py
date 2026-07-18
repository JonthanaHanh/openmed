"""Per-language locale + national-ID surrogate coherence suite (OM-135).

A regression guard that keeps language packs honest as the count climbs from
12 toward 20+. It asserts that:

  (a) every ``SUPPORTED_LANGUAGES`` code maps to a real Faker locale (or a
      documented approximation),
  (b) generated national-ID surrogates pass the language's registered
      validator (round-trip surrogate fidelity),
  (c) only documented approximations emit the locale ``UserWarning`` — so a new
      pack mis-wired to a wrong locale fails loudly, and
  (d) ``locale_coherence_report()`` stays a faithful per-language summary the
      status/leaderboard work can reuse.

The contract these tests gate lives in
:mod:`openmed.core.anonymizer.locales`.
"""

import json
import warnings

import pytest
from faker import Faker
from faker.config import AVAILABLE_LOCALES

from openmed.core.anonymizer import locales as L
from openmed.core.anonymizer.locales import (
    FAKER_BACKEND_LOCALE,
    LANG_TO_LOCALE,
    NATIONAL_ID_PROVIDERS,
    locale_coherence_report,
    resolve_locale,
)
from openmed.core.anonymizer.providers.clinical_ids import register_clinical_providers
from openmed.core.anonymizer.providers.registry_ids import get_national_id
from openmed.core.anonymizer.registry import _LOCALE_ID_METHODS
from openmed.core.pii_entity_merger import PII_PATTERNS
from openmed.core.pii_i18n import (
    LANGUAGE_PII_PATTERNS,
    NATIONAL_ID_ONLY_LANGUAGES,
    SUPPORTED_LANGUAGES,
)

# Documented set of languages whose *default* Faker locale is an intentional
# approximation. Kept here, independent of the code under test, so that wiring a
# new pack to a wrong/approximate locale flips the assertions red until a human
# consciously updates this set.
DOCUMENTED_APPROXIMATE = {"te", "ms", "sr"}

# Languages that register a national-ID validator but have no Faker
# surrogate provider yet. Documented so a *new* such gap can't slip in silently.
KNOWN_PROVIDERLESS_VALIDATORS = set()

# Number of surrogates sampled per language for the round-trip check.
SAMPLE_SIZE = 40

REPORT_LANGUAGES = SUPPORTED_LANGUAGES | NATIONAL_ID_ONLY_LANGUAGES


def _languages_with_national_id_validator():
    return {
        lang
        for lang in REPORT_LANGUAGES
        if any(
            p.validator is not None and p.entity_type == "national_id"
            for p in LANGUAGE_PII_PATTERNS.get(lang, [])
        )
    }


def _national_id_validators(lang):
    """Registered national-ID checksum validators for ``lang``.

    Prefer the language pack's own ``national_id`` validators; fall back to the
    shared base SSN/national-ID validators for languages (e.g. English) whose
    national ID is validated by the base pattern set.
    """
    lang_validators = [
        p.validator
        for p in LANGUAGE_PII_PATTERNS.get(lang, [])
        if p.validator is not None and p.entity_type == "national_id"
    ]
    if lang_validators:
        return lang_validators
    return [
        p.validator
        for p in PII_PATTERNS
        if p.validator is not None and p.entity_type in ("national_id", "ssn")
    ]


class TestLocaleResolution:
    @pytest.mark.parametrize("lang", sorted(REPORT_LANGUAGES))
    def test_every_supported_language_has_locale_entry(self, lang):
        assert lang in LANG_TO_LOCALE, f"{lang!r} missing LANG_TO_LOCALE entry"

    @pytest.mark.parametrize("lang", sorted(REPORT_LANGUAGES))
    def test_resolved_locale_exists_in_faker(self, lang):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            locale = resolve_locale(lang)
        assert locale, f"{lang!r} resolved to an empty locale"
        assert locale in AVAILABLE_LOCALES or lang in L._APPROXIMATE_LOCALES, (
            f"{lang!r} -> {locale!r} is not a real Faker locale and is not a "
            "documented approximation"
        )


class TestNationalIdRoundTrip:
    def test_contract_matches_registry_dispatch(self):
        """Each provider's method must match the registry's locale dispatch and
        point at a real Faker locale."""
        for lang, providers in NATIONAL_ID_PROVIDERS.items():
            for index, (id_type, (locale, method)) in enumerate(providers.items()):
                backend_locale = FAKER_BACKEND_LOCALE.get(locale, locale)
                assert backend_locale in AVAILABLE_LOCALES, (
                    f"{lang!r}/{id_type!r} -> unknown Faker backend locale "
                    f"{backend_locale!r}"
                )
                if index == 0:
                    assert _LOCALE_ID_METHODS.get(locale) == method, (
                        f"{lang!r} primary provider {method!r} disagrees with "
                        f"registry dispatch for {locale!r} "
                        f"({_LOCALE_ID_METHODS.get(locale)!r})"
                    )
                spec = get_national_id(lang, id_type)
                if spec is not None:
                    assert spec.faker_method == method

    def test_providerless_validators_are_documented(self):
        """A national-ID validator with no surrogate provider is a known gap;
        a new one must be acknowledged here rather than slip in silently."""
        without_provider = _languages_with_national_id_validator() - set(
            NATIONAL_ID_PROVIDERS
        )
        assert without_provider == KNOWN_PROVIDERLESS_VALIDATORS, (
            "languages with a national_id validator but no NATIONAL_ID_PROVIDERS "
            f"entry changed: {sorted(without_provider)}"
        )

    @pytest.mark.parametrize("lang", sorted(NATIONAL_ID_PROVIDERS))
    def test_generated_national_ids_pass_registered_validator(self, lang):
        for id_type, (locale, method) in NATIONAL_ID_PROVIDERS[lang].items():
            spec = get_national_id(lang, id_type)
            validators = (
                [spec.validate] if spec is not None else _national_id_validators(lang)
            )
            assert validators, (
                f"no national-ID validator registered for {lang!r}/{id_type!r}"
            )
            faker = Faker(FAKER_BACKEND_LOCALE.get(locale, locale))
            register_clinical_providers(faker)
            for seed in range(SAMPLE_SIZE):
                faker.seed_instance(seed)
                surrogate = getattr(faker, method)()
                assert any(validator(surrogate) for validator in validators), (
                    f"{lang!r}/{id_type!r} surrogate {surrogate!r} "
                    f"(seed={seed}, locale={locale}) failed every registered "
                    f"validator {[validator.__name__ for validator in validators]}"
                )


class TestApproximateLocaleWarnings:
    @pytest.mark.parametrize("lang", sorted(REPORT_LANGUAGES))
    def test_only_documented_approximations_warn(self, lang):
        L._warned.clear()  # reset the one-time-warning cache for a clean read
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resolve_locale(lang)
        warned = any(issubclass(w.category, UserWarning) for w in caught)
        assert warned == (lang in DOCUMENTED_APPROXIMATE), (
            f"{lang!r}: emitted approximate-locale warning={warned}, "
            f"expected {lang in DOCUMENTED_APPROXIMATE}"
        )

    def test_code_approximation_set_matches_documentation(self):
        assert set(L._APPROXIMATE_LOCALES) == DOCUMENTED_APPROXIMATE


class TestLocaleCoherenceReport:
    def test_one_row_per_supported_language(self):
        rows = locale_coherence_report()
        assert len(rows) == len(REPORT_LANGUAGES)
        assert {r["language"] for r in rows} == REPORT_LANGUAGES

    def test_row_shape_and_values(self):
        rows = {r["language"]: r for r in locale_coherence_report()}
        for lang, row in rows.items():
            assert set(row) == {
                "language",
                "locale",
                "approximate",
                "id_providers",
                "id_types",
                "id_locale",
            }
            assert row["locale"] == LANG_TO_LOCALE[lang]
            assert isinstance(row["approximate"], bool)
            assert row["approximate"] == (lang in DOCUMENTED_APPROXIMATE)
            assert isinstance(row["id_providers"], list)
            if lang in NATIONAL_ID_PROVIDERS:
                providers = NATIONAL_ID_PROVIDERS[lang]
                assert row["id_types"] == list(providers)
                assert row["id_providers"] == [
                    method for _locale, method in providers.values()
                ]
                assert row["id_locale"] == next(
                    iter({locale for locale, _method in providers.values()})
                )
            else:
                assert row["id_providers"] == []
                assert row["id_types"] == []
                assert row["id_locale"] is None

    def test_report_is_json_serializable(self):
        # The status/leaderboard work serializes this; keep it JSON-friendly.
        json.dumps(locale_coherence_report())
