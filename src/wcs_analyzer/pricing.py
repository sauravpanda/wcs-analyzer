"""Per-model API pricing and cost estimation.

The published prices in PRICING are a point-in-time snapshot — provider
pricing changes more often than this package gets released, so the
estimate is best-effort. `PRICING_UPDATED_ON` is surfaced in every
report alongside the cost so users know what the numbers are based on.

To override the table without editing source, set WCS_PRICING_FILE to a
JSON file shaped like:

    {
      "updated_on": "2026-06-01",
      "models": {
        "gemini-2.5-flash": {"input_per_mtok": 0.30, "output_per_mtok": 2.50}
      }
    }

Only listed models are overridden — any model absent from the override
file falls back to the bundled defaults.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


PRICING_UPDATED_ON = "2026-04-15"


@dataclass
class ModelPricing:
    """USD cost per million tokens for a given model."""

    input_per_mtok: float
    output_per_mtok: float


# Prices in USD per 1M tokens (text). Video / audio / image tokens on
# Gemini are typically billed at the same rate as the closest modality,
# and total_token_count already sums them, so we use a single rate per
# model here. Update PRICING_UPDATED_ON whenever this table moves.
_DEFAULT_PRICING: dict[str, ModelPricing] = {
    # Google Gemini
    "gemini-2.5-flash": ModelPricing(0.30, 2.50),
    "gemini-2.5-pro": ModelPricing(1.25, 10.00),
    "gemini-1.5-flash": ModelPricing(0.075, 0.30),
    "gemini-1.5-pro": ModelPricing(1.25, 5.00),
    # Anthropic Claude (4.x generation)
    "claude-opus-4-6": ModelPricing(15.00, 75.00),
    "claude-sonnet-4-6": ModelPricing(3.00, 15.00),
    "claude-haiku-4-5-20251001": ModelPricing(1.00, 5.00),
    # Legacy fallbacks
    "claude-sonnet-4-5": ModelPricing(3.00, 15.00),
    "claude-opus-4-5": ModelPricing(15.00, 75.00),
}


def _load_override() -> tuple[dict[str, ModelPricing], str | None]:
    """Read WCS_PRICING_FILE if set; return (overrides, updated_on)."""
    path_str = os.environ.get("WCS_PRICING_FILE")
    if not path_str:
        return {}, None
    path = Path(path_str)
    if not path.exists():
        logger.warning("WCS_PRICING_FILE %s does not exist; ignoring", path)
        return {}, None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse WCS_PRICING_FILE %s: %s", path, e)
        return {}, None
    overrides: dict[str, ModelPricing] = {}
    for model, entry in (data.get("models") or {}).items():
        try:
            overrides[model] = ModelPricing(
                input_per_mtok=float(entry["input_per_mtok"]),
                output_per_mtok=float(entry["output_per_mtok"]),
            )
        except (KeyError, TypeError, ValueError):
            logger.warning("Bad pricing entry for %s in override file", model)
    return overrides, data.get("updated_on")


def get_pricing(model: str) -> ModelPricing | None:
    """Return pricing for a model, consulting WCS_PRICING_FILE first."""
    overrides, _ = _load_override()
    if model in overrides:
        return overrides[model]
    return _DEFAULT_PRICING.get(model)


def pricing_updated_on() -> str:
    """Return the effective 'updated on' date for the current pricing."""
    _, override_date = _load_override()
    return override_date or PRICING_UPDATED_ON


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return estimated USD cost for the given token counts.

    Returns 0.0 (rather than raising) when the model isn't in the price
    table — the caller should treat unknown models as "cost unknown",
    not "free". The report layer flags this case.
    """
    p = get_pricing(model)
    if p is None:
        return 0.0
    return (
        (input_tokens / 1_000_000) * p.input_per_mtok
        + (output_tokens / 1_000_000) * p.output_per_mtok
    )


@dataclass
class UsageTotals:
    """Aggregated token counts and cost for a run."""

    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost: float = 0.0
    model: str = ""
    pricing_known: bool = False

    @classmethod
    def from_counts(cls, model: str, input_tokens: int, output_tokens: int) -> "UsageTotals":
        cost = estimate_cost(model, input_tokens, output_tokens)
        return cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost=round(cost, 4),
            model=model,
            pricing_known=get_pricing(model) is not None,
        )

    def add(self, other: "UsageTotals") -> "UsageTotals":
        """Return a new UsageTotals summing self and other.

        If the models differ (ensemble mode), keeps the current model
        name and marks pricing_known only if both sides were known.
        """
        return UsageTotals(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            estimated_cost=round(self.estimated_cost + other.estimated_cost, 4),
            model=self.model or other.model,
            pricing_known=self.pricing_known and other.pricing_known,
        )
