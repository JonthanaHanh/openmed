"""Tests for the locale national-ID provider registry."""

from __future__ import annotations

import pytest
from faker import Faker

from openmed.core.anonymizer.locales import FAKER_BACKEND_LOCALE
from openmed.core.anonymizer.providers import registry_ids
from openmed.core.anonymizer.providers.clinical_ids import (
    AfricanPhoneProvider,
    register_clinical_providers,
)
from openmed.core.anonymizer.providers.registry_ids import (
    ID_PROVIDER_REGISTRY,
    NationalIdSpec,
    clinical_faker_provider_classes,
    get_national_id,
    register_national_id,
)


def _always_valid(_text: str) -> bool:
    return True


EXPECTED_VALIDATOR_KEYS = (
    ("fr", "nir"),
    ("de", "steuer_id"),
    ("it", "codice_fiscale"),
    ("es", "dni"),
    ("es", "nie"),
    ("nl", "bsn"),
    ("in", "aadhaar"),
    ("id", "nik"),
    ("ms", "mykad"),
    ("tl", "philsys_psn"),
    ("tl", "philhealth_pin"),
    ("da", "cpr"),
    ("th", "thai_national_id"),
    ("he", "teudat_zehut"),
    ("pl", "pesel"),
    ("lv", "personas_kods"),
    ("ko", "rrn"),
    ("hu", "taj"),
    ("cs", "rodne_cislo"),
    ("sk", "rodne_cislo"),
    ("pt", "cpf"),
    ("pt", "cnpj"),
    ("tr", "tckn"),
    ("vi", "cccd"),
    ("vi", "cmnd"),
    ("us", "npi"),
)


ROUND_TRIP_CASES = (
    ("fr", "nir", "fr_FR"),
    ("de", "steuer_id", "de_DE"),
    ("it", "codice_fiscale", "it_IT"),
    ("es", "dni", "es_ES"),
    ("es", "nie", "es_ES"),
    ("nl", "bsn", "nl_NL"),
    ("in", "aadhaar", "en_IN"),
    ("id", "nik", "id_ID"),
    ("ms", "mykad", "ms_MY"),
    ("tl", "philsys_psn", "fil_PH"),
    ("tl", "philhealth_pin", "fil_PH"),
    ("da", "cpr", "da_DK"),
    ("th", "thai_national_id", "th_TH"),
    ("he", "teudat_zehut", "he_IL"),
    ("pl", "pesel", "pl_PL"),
    ("lv", "personas_kods", "lv_LV"),
    ("ko", "rrn", "ko_KR"),
    ("hu", "taj", "hu_HU"),
    ("cs", "rodne_cislo", "cs_CZ"),
    ("sk", "rodne_cislo", "sk_SK"),
    ("pt", "cpf", "pt_BR"),
    ("pt", "cnpj", "pt_BR"),
    ("tr", "tckn", "tr_TR"),
    ("vi", "cccd", "vi_VN"),
    ("vi", "cmnd", "vi_VN"),
    ("us", "npi", "en_US"),
)


class TestNationalIdRegistry:
    def test_acceptance_lookups_return_validating_specs(self):
        for lang, id_type, locale in (
            ("it", "codice_fiscale", "it_IT"),
            ("in", "aadhaar", "en_IN"),
        ):
            spec = get_national_id(lang, id_type)
            assert spec is not None

            faker = Faker(locale)
            register_clinical_providers(faker)
            surrogate = getattr(faker, spec.faker_method)()

            assert spec.validate(surrogate), (
                f"{lang!r}/{id_type!r} generated invalid surrogate {surrogate!r}"
            )

    @pytest.mark.parametrize(("lang", "id_type"), EXPECTED_VALIDATOR_KEYS)
    def test_pre_existing_validators_are_reachable(self, lang, id_type):
        spec = get_national_id(lang, id_type)
        assert spec is not None
        assert spec.validate is not None
        assert spec.faker_method

    @pytest.mark.parametrize(("lang", "id_type", "locale"), ROUND_TRIP_CASES)
    def test_registered_surrogates_pass_registered_validator(
        self,
        lang,
        id_type,
        locale,
    ):
        spec = get_national_id(lang, id_type)
        assert spec is not None

        faker = Faker(FAKER_BACKEND_LOCALE.get(locale, locale))
        register_clinical_providers(faker)
        faker.seed_instance(42)
        surrogate = getattr(faker, spec.faker_method)()

        assert spec.validate(surrogate), (
            f"{lang!r}/{id_type!r} generated invalid surrogate {surrogate!r}"
        )

    def test_lookup_normalizes_case_hyphens_and_locale(self):
        assert get_national_id("IT-it", "Codice Fiscale") == get_national_id(
            "it_IT",
            "codice_fiscale",
        )
        assert get_national_id("EN-in", "Aadhaar") == get_national_id(
            "en_IN",
            "aadhaar",
        )

    def test_unknown_lookup_returns_none(self):
        assert get_national_id("zz", "unknown") is None

    def test_duplicate_registration_rejects_key_with_clear_error(self):
        key = ("zz", "dummy")
        ID_PROVIDER_REGISTRY.pop(key, None)
        spec = NationalIdSpec("zz", "dummy", _always_valid, "word")

        try:
            register_national_id(spec)
            with pytest.raises(ValueError, match="already registered.*zz.*dummy"):
                register_national_id(spec)
        finally:
            ID_PROVIDER_REGISTRY.pop(key, None)

    def test_module_docstring_documents_language_pack_steps(self):
        doc = registry_ids.__doc__
        assert doc is not None
        for expected in (
            "Implement or import a deterministic validator",
            "Ensure Faker can generate matching surrogates",
            "Register a NationalIdSpec",
            "Register every needed alias explicitly",
            "Add unit tests",
        ):
            assert expected in doc


class TestAfricanPhoneProviderRegistry:
    def test_provider_is_registered_once(self):
        provider_classes = clinical_faker_provider_classes()

        assert provider_classes.count(AfricanPhoneProvider) == 1

    @pytest.mark.parametrize(
        ("original", "preserved_digits"),
        [
            ("+251 91 234 5678", "25191"),
            ("00250 78 123 4567", "0025078"),
            ("0752 876 543", "0752"),
        ],
    )
    def test_registered_provider_preserves_country_and_operator_prefix(
        self,
        original,
        preserved_digits,
    ):
        faker = Faker("en_US")
        register_clinical_providers(faker)
        faker.seed_instance(858)

        surrogate = faker.african_phone(original)

        assert surrogate is not None
        assert "".join(char for char in surrogate if char.isdigit()).startswith(
            preserved_digits
        )
        assert surrogate != original
        assert "".join(char for char in surrogate if not char.isdigit()) == "".join(
            char for char in original if not char.isdigit()
        )

    def test_seeded_provider_is_deterministic(self):
        first = Faker("en_US")
        second = Faker("en_US")
        register_clinical_providers(first)
        register_clinical_providers(second)
        first.seed_instance(858)
        second.seed_instance(858)

        assert first.african_phone("+233 24 123 4567") == second.african_phone(
            "+233 24 123 4567"
        )
