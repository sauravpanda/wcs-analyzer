"""Tests for the pricing / cost-estimation module."""

import json
from pathlib import Path

import pytest

from wcs_analyzer.pricing import (
    PRICING_UPDATED_ON,
    UsageTotals,
    estimate_cost,
    get_pricing,
    pricing_updated_on,
)


class TestEstimateCost:
    def test_known_model_cost(self):
        # Gemini 2.5 Flash: $0.30/M input, $2.50/M output
        cost = estimate_cost("gemini-2.5-flash", input_tokens=1_000_000, output_tokens=1_000_000)
        assert cost == pytest.approx(0.30 + 2.50)

    def test_fractional_tokens(self):
        # Sonnet 4.6: $3/M input, $15/M output
        cost = estimate_cost("claude-sonnet-4-6", input_tokens=100_000, output_tokens=50_000)
        assert cost == pytest.approx(0.30 + 0.75)

    def test_unknown_model_returns_zero(self):
        assert estimate_cost("unknown-model", 1000, 500) == 0.0

    def test_zero_tokens_is_zero_cost(self):
        assert estimate_cost("gemini-2.5-flash", 0, 0) == 0.0


class TestGetPricing:
    def test_known_model(self):
        p = get_pricing("gemini-2.5-flash")
        assert p is not None
        assert p.input_per_mtok > 0
        assert p.output_per_mtok > 0

    def test_unknown_model(self):
        assert get_pricing("bogus-model") is None

    def test_default_models_have_pricing(self):
        """Every provider's default model must be in the price table.

        Guards against future drift where we bump the default but
        forget to add a pricing entry — the analyzer would then report
        `$0.00 (unknown model)` on every run.
        """
        from wcs_analyzer.cli import _DEFAULT_MODELS
        for provider, model in _DEFAULT_MODELS.items():
            p = get_pricing(model)
            assert p is not None, f"Default model for {provider} ({model}) has no pricing entry"

    def test_gemini_3_pro_pricing(self):
        p = get_pricing("gemini-3.1-pro-preview")
        assert p is not None
        # Standard-tier published prices as of 2026-04: $2 / $12
        assert p.input_per_mtok == 2.00
        assert p.output_per_mtok == 12.00


class TestPricingOverride:
    def test_env_override_replaces_defaults(self, tmp_path: Path, monkeypatch):
        override = {
            "updated_on": "2030-01-01",
            "models": {
                "gemini-2.5-flash": {"input_per_mtok": 99.0, "output_per_mtok": 100.0}
            }
        }
        override_file = tmp_path / "pricing.json"
        override_file.write_text(json.dumps(override))
        monkeypatch.setenv("WCS_PRICING_FILE", str(override_file))

        p = get_pricing("gemini-2.5-flash")
        assert p is not None
        assert p.input_per_mtok == 99.0
        assert p.output_per_mtok == 100.0
        # Non-overridden models fall back to defaults
        sonnet = get_pricing("claude-sonnet-4-6")
        assert sonnet is not None
        assert sonnet.input_per_mtok == 3.00
        # Updated-on date reflects the override
        assert pricing_updated_on() == "2030-01-01"

    def test_missing_override_file_logs_and_falls_back(self, monkeypatch, tmp_path: Path):
        monkeypatch.setenv("WCS_PRICING_FILE", str(tmp_path / "does_not_exist.json"))
        assert get_pricing("gemini-2.5-flash") is not None
        assert pricing_updated_on() == PRICING_UPDATED_ON

    def test_malformed_override_file_falls_back(self, monkeypatch, tmp_path: Path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json")
        monkeypatch.setenv("WCS_PRICING_FILE", str(bad))
        assert get_pricing("gemini-2.5-flash") is not None

    def test_bad_entry_in_override_skipped(self, monkeypatch, tmp_path: Path):
        override = {"models": {"gemini-2.5-flash": {"input_per_mtok": "not-a-number"}}}
        bad = tmp_path / "partial.json"
        bad.write_text(json.dumps(override))
        monkeypatch.setenv("WCS_PRICING_FILE", str(bad))
        # Bad entry dropped -> falls back to default
        p = get_pricing("gemini-2.5-flash")
        assert p is not None
        assert p.input_per_mtok == 0.30


class TestUsageTotals:
    def test_from_counts_computes_cost(self):
        u = UsageTotals.from_counts("claude-sonnet-4-6", 10_000, 5_000)
        assert u.input_tokens == 10_000
        assert u.output_tokens == 5_000
        assert u.estimated_cost > 0
        assert u.model == "claude-sonnet-4-6"
        assert u.pricing_known is True

    def test_from_counts_unknown_model(self):
        u = UsageTotals.from_counts("made-up-model", 1000, 500)
        assert u.estimated_cost == 0.0
        assert u.pricing_known is False

    def test_add_sums_tokens_and_cost(self):
        a = UsageTotals.from_counts("claude-sonnet-4-6", 1000, 500)
        b = UsageTotals.from_counts("claude-sonnet-4-6", 2000, 1000)
        total = a.add(b)
        assert total.input_tokens == 3000
        assert total.output_tokens == 1500
        assert total.estimated_cost == pytest.approx(a.estimated_cost + b.estimated_cost)

    def test_add_empty_preserves_self(self):
        a = UsageTotals.from_counts("gemini-2.5-flash", 1000, 500)
        empty = UsageTotals()
        total = a.add(empty)
        assert total.input_tokens == 1000
        assert total.output_tokens == 500
        assert total.model == "gemini-2.5-flash"

    def test_add_mixed_pricing_known_false(self):
        a = UsageTotals.from_counts("claude-sonnet-4-6", 1000, 500)
        b = UsageTotals.from_counts("unknown-model", 2000, 1000)
        total = a.add(b)
        assert total.pricing_known is False
