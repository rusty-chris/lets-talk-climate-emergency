"""Energy/carbon footprint estimation — the OWNER-APPROVED methodology as code.

RED-phase contract stubs: behaviour raises ``NotImplementedError``; the
failing suites in ``tests/unit/test_footprint_estimation.py``,
``tests/unit/test_footprint_ledger.py``,
``tests/unit/test_footprint_page.py``,
``tests/unit/test_service_footprint_route.py`` and
``tests/unit/test_ui_footprint.py`` pin the contract.

**Source of truth:** ``docs/FOOTPRINT-METHODOLOGY.md`` (owner-approved,
PR #355) is BINDING. Every factor constant below is pinned EQUAL to that
document's §3 table by a docs-as-code parity test, and every verbatim
sentence constant is pinned present in the document — the code can never
drift from the published method. A factor change is therefore a visible,
dated event: doc + constant + :data:`FOOTPRINT_FACTORS_VERSION` move in
one commit or the parity suite fails.

**The register contract (methodology §9):** every figure is labelled
measured, estimated, or unknown; estimates are ALWAYS ranges (low/high
propagated through the low/high ends of every factor — never a point
estimate with decoration); the footer indicator carries no gCO2e (the
carbon figure adds the grid assumption and belongs on the /footprint
page where that assumption is disclosed beside it).

**The $0 rule:** everything in this module is local arithmetic over
token counts and CPU-seconds already in hand. This module makes NO
network calls and imports NO adapter, provider SDK, HTTP client or web
framework — pinned structurally by the estimation suite. Serving the
/footprint page and computing every figure on it must cost $0 and work
identically in the paused state.

Layout (module-home DECISION, flagged in the red-phase report): this
single module holds (a) the pure estimation model, (b) the verbatim
presentation constants, and (c) :class:`FootprintLedger`, the
privacy-safe persistent aggregate. It lives in ``service/`` because the
ledger writes beside the #217 spend journal, but it stays PURE
(stdlib + ``service.atomic_write`` only) so ``ui.render_model`` can
import the estimation functions directly — the same one-way
ui-imports-pure-service direction as ``service.exchange_log``. One
source of truth for the factors; no duplicated constants to parity-pin.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from service.atomic_write import atomic_write_text

__all__ = [
    "EnergyFactor",
    "WhRange",
    "GramsCO2eRange",
    "E_OUT_WH_PER_1K",
    "E_IN_WH_PER_1K",
    "CACHE_READ_FACTOR",
    "CIF_API_G_PER_KWH",
    "W_PER_VCPU",
    "PUE_HETZNER",
    "CIF_LOCAL_DE_G_PER_KWH",
    "CIF_LOCAL_FI_G_PER_KWH",
    "CIF_LOCAL_G_PER_KWH",
    "SONNET_ENERGY_MULTIPLIER",
    "OPUS_ENERGY_MULTIPLIER_LOW",
    "OPUS_ENERGY_MULTIPLIER_HIGH",
    "STREAMING_G_CO2E_PER_HOUR",
    "CAR_G_CO2E_PER_METRE",
    "TEA_MUG_WH",
    "FOOTPRINT_ROUTE",
    "FOOTPRINT_FACTORS_VERSION",
    "FOOTPRINT_FOOTER_TEMPLATE",
    "FOOTPRINT_FOOTER_CACHED_TEMPLATE",
    "FOOTPRINT_ESTIMATES_PHRASE",
    "ANTHROPIC_UNCERTAINTY_PARAGRAPH",
    "MARKET_VS_LOCATION_SENTENCE",
    "ANTHROPIC_GRID_ASSUMPTION_SENTENCE",
    "TRAINING_EXCLUSION_PHRASE",
    "FOOTPRINT_TOTALS_UNAVAILABLE_NOTICE",
    "FOOTPRINT_STATE_FILENAME",
    "FOOTPRINT_JOURNAL_ALLOWED_KEYS",
    "FootprintTotals",
    "FootprintLedgerError",
    "FootprintLedger",
    "api_energy_wh",
    "local_energy_wh",
    "co2e_grams",
    "sum_wh_ranges",
    "usage_token_counts",
    "streaming_seconds_equivalent",
    "metres_driven_equivalent",
    "exchanges_per_mug_of_tea",
    "format_wh_value",
    "format_wh_bound",
    "format_footprint_footer",
    "format_footprint_footer_cached",
]


# ---------------------------------------------------------------------------
# The §3 factor table (docs-as-code: pinned equal to the methodology doc)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnergyFactor:
    """One sourced, ranged conversion factor: low/central/high bounds.

    The range is the honesty mechanism: bounds propagate through the §2
    formula with the low/high ends of EVERY factor, so a displayed range
    is a real propagation, never a point estimate with decoration.
    """

    low: float
    central: float
    high: float

    def __post_init__(self) -> None:
        if not (self.low <= self.central <= self.high):
            raise ValueError(
                f"EnergyFactor bounds out of order: {self.low} / {self.central} / {self.high}"
            )


#: Wh per 1k OUTPUT tokens, Haiku-class (methodology §3, facility energy).
E_OUT_WH_PER_1K = EnergyFactor(low=0.1, central=0.5, high=1.5)

#: Wh per 1k INPUT tokens, incl. cache writes (methodology §3).
E_IN_WH_PER_1K = EnergyFactor(low=0.005, central=0.02, high=0.07)

#: Cache-read token energy as a fraction of an uncached input token
#: (§3). Explicitly NOT Anthropic's 0.1× billing ratio — a price is not
#: an energy measurement.
CACHE_READ_FACTOR = EnergyFactor(low=0.05, central=0.08, high=0.20)

#: Grid carbon intensity assumed for Anthropic serving, gCO2e/kWh (§3/§4:
#: US average, location-based, region unknown and disclosed as such).
CIF_API_G_PER_KWH = EnergyFactor(low=287.0, central=384.0, high=450.0)

#: Watts per shared cloud vCPU at typical utilisation (§3, Cloud Carbon
#: Footprint coefficients).
W_PER_VCPU = EnergyFactor(low=0.7, central=2.2, high=3.8)

#: Hetzner's supplier-published average PUE (§3).
PUE_HETZNER = EnergyFactor(low=1.1, central=1.13, high=1.2)

#: Location-based grid factors for the Hetzner DCs, gCO2e/kWh (§3/§4).
CIF_LOCAL_DE_G_PER_KWH = 363.0
CIF_LOCAL_FI_G_PER_KWH = 95.0

#: The factor for the DC actually chosen. DECISION (flagged in the
#: red-phase report): the production runbook targets a German-region
#: Hetzner box, so the German location-based factor applies — the
#: HIGHER, more honest figure (§4: we do not net off the market-based
#: renewable claim).
CIF_LOCAL_G_PER_KWH = CIF_LOCAL_DE_G_PER_KWH

#: §3.4 non-default models: Sonnet bake-off arm ×2; Opus best mode ×3–4
#: on BOTH token factors (flagged in the doc as the weakest numbers in
#: the table — labelled "extrapolated" wherever they surface).
SONNET_ENERGY_MULTIPLIER = 2.0
OPUS_ENERGY_MULTIPLIER_LOW = 3.0
OPUS_ENERGY_MULTIPLIER_HIGH = 4.0

#: §8 anchor factors, each with a citable source (IEA/Kamiya 2020; US
#: EPA equivalencies calculator; first-principles kettle physics at the
#: stated ~80% efficiency — 0.031 kWh per 250 ml mug).
STREAMING_G_CO2E_PER_HOUR = 36.0
CAR_G_CO2E_PER_METRE = 0.25
TEA_MUG_WH = 31.0

#: The transparency route this feature adds (parity-pinned into both
#: ``service.transparency.TRANSPARENCY_ROUTES`` and
#: ``ui.footer.TRANSPARENCY_ROUTES``).
FOOTPRINT_ROUTE = "/footprint"

#: §9.8 revision note: the factors are versioned in the repo and the
#: /footprint page shows this version/date, so a factor update is a
#: visible, dated event. Bump WITH any factor change.
FOOTPRINT_FACTORS_VERSION = "2026-09 (v1)"


# ---------------------------------------------------------------------------
# Verbatim presentation constants (§9 / §4 — pinned present in the doc)
# ---------------------------------------------------------------------------

#: The footer indicator, VERBATIM from methodology §9. Rules the tests
#: enforce: always a range, never a point; "est."/"estimates" always
#: present; the /footprint link always present; NO gCO2e in the footer;
#: Wh (not J or kWh).
FOOTPRINT_FOOTER_TEMPLATE = (
    "Footprint: est. {lo}–{hi} Wh this answer · est. {session_lo}–{session_hi} "
    "Wh this session — estimates, not measurements · how we know → /footprint"
)

#: The register phrase the indicator must always carry (§9).
FOOTPRINT_ESTIMATES_PHRASE = "estimates, not measurements"

#: The cached-exchange indicator (DECISION, flagged in the red-phase
#: report): a semantic-cache or cached-starter replay performs ~zero new
#: inference — the honest per-answer figure is "~0 Wh (cached)", never a
#: fabricated estimate range, and it contributes zero to the session
#: accumulation. Keeps every §9 register rule: the estimates phrase, the
#: /footprint link, no gCO2e, session still a range.
FOOTPRINT_FOOTER_CACHED_TEMPLATE = (
    "Footprint: ~0 Wh this answer (cached — no new model inference) · "
    "est. {session_lo}–{session_hi} Wh this session — estimates, not "
    "measurements · how we know → /footprint"
)

#: §9 page structure item 4, VERBATIM on the /footprint page.
ANTHROPIC_UNCERTAINTY_PARAGRAPH = (
    "Anthropic does not publish the energy its models use. Our per-token "
    "factor spans 15×; the true figure is somewhere in that range, and we "
    "will not pretend to know where. If Anthropic publishes measurements, "
    "we will replace this section with them."
)

#: §4's one-sentence Hetzner market-vs-location treatment, VERBATIM.
MARKET_VS_LOCATION_SENTENCE = (
    "Market-based accounting (counting our host's certified renewable "
    "purchases) puts our own server near zero gCO2e; location-based "
    "accounting (the average grid where the server physically draws power) "
    "puts it at ~363 gCO2e/kWh in Germany — we show the location-based "
    "figure and note the renewable claim."
)

#: §4's Anthropic grid assumption, VERBATIM on the page.
ANTHROPIC_GRID_ASSUMPTION_SENTENCE = (
    "We do not know where Anthropic serves our requests; we assume the US "
    "average grid. Cloud providers' market-based renewable claims would "
    "reduce this on paper; we report the location-based figure."
)

#: §7: model training is excluded and the page says so in these words.
TRAINING_EXCLUSION_PHRASE = "not included, not zero"

#: The honest state when the aggregate journal cannot be read: the page
#: says the totals are unavailable — it NEVER silently renders zeros
#: (an undercount presented as truth) and never fabricates.
FOOTPRINT_TOTALS_UNAVAILABLE_NOTICE = (
    "Our running totals are temporarily unavailable (the counter file "
    "could not be read). Nothing is estimated in their place — the "
    "methodology below still applies to every answer."
)


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WhRange:
    """An estimated energy figure in Wh: an honest low/central/high range."""

    low: float
    central: float
    high: float


@dataclass(frozen=True)
class GramsCO2eRange:
    """An estimated carbon figure in grams CO2e: low/central/high."""

    low: float
    central: float
    high: float


#: A zero range — the additive identity for session/application sums.
WH_ZERO = WhRange(low=0.0, central=0.0, high=0.0)


# ---------------------------------------------------------------------------
# The §2 estimation model (pure arithmetic — stubs, pinned RED)
# ---------------------------------------------------------------------------


def usage_token_counts(usage: Mapping[str, Any]) -> dict[str, int]:
    """Normalise one adapter usage mapping to the four token-count keys.

    RED-phase contract stub; ``tests/unit/test_footprint_estimation.py``
    pins: accepts the seam's Anthropic key names (``input_tokens``,
    ``output_tokens``, ``cache_read_input_tokens``,
    ``cache_creation_input_tokens``); ``None``/absent keys count as zero
    (the ``service.budget`` convention); unknown keys are ignored;
    result always carries exactly the four keys as ints.
    """

    def count(key: str) -> int:
        value = usage.get(key)
        # The service.budget convention: None/absent/zero all fold to 0.
        return int(value) if value else 0

    return {
        "input_tokens": count("input_tokens"),
        "output_tokens": count("output_tokens"),
        "cache_read_input_tokens": count("cache_read_input_tokens"),
        "cache_creation_input_tokens": count("cache_creation_input_tokens"),
    }


#: The energy factors and family multipliers as (low, central, high) end
#: names, so the like-ends-together propagation loops once per bound.
_FACTOR_ENDS: tuple[str, ...] = ("low", "central", "high")


def _model_multipliers(model: str | None) -> tuple[float, float, float]:
    """The §3.4 per-bound family multipliers for ``model`` (prefix rule).

    ``None`` / ``claude-haiku*`` ×1 on every bound; ``claude-sonnet*``
    ×2; ``claude-opus*`` widens the range — ×3 on the low bound, ×4 on
    the high, the mid-multiplier on the central (the doc flags these as
    extrapolated). An unrecognised family refuses loudly (the pricing
    seam's unknown-model rule — never a silently wrong factor).
    """
    if model is None or model.startswith("claude-haiku"):
        return (1.0, 1.0, 1.0)
    if model.startswith("claude-sonnet"):
        return (
            SONNET_ENERGY_MULTIPLIER,
            SONNET_ENERGY_MULTIPLIER,
            SONNET_ENERGY_MULTIPLIER,
        )
    if model.startswith("claude-opus"):
        return (
            OPUS_ENERGY_MULTIPLIER_LOW,
            (OPUS_ENERGY_MULTIPLIER_LOW + OPUS_ENERGY_MULTIPLIER_HIGH) / 2,
            OPUS_ENERGY_MULTIPLIER_HIGH,
        )
    raise ValueError(
        f"unknown model family for footprint estimation: {model!r} — "
        "refusing a silently-wrong energy factor (§3.4)"
    )


def api_energy_wh(
    usage_mappings: Iterable[Mapping[str, Any]],
    *,
    model: str | None = None,
) -> WhRange:
    """The §2 ``E_api`` term over one exchange's usage mappings → Wh range.

    RED-phase contract stub; the failing suite pins:

    - per mapping: ``input/1000×E_IN + cache_creation/1000×E_IN +
      cache_read/1000×E_IN×CACHE_READ_FACTOR + output/1000×E_OUT``,
      each bound computed with the SAME end of every factor (low with
      low, high with high) — honest propagation, never a point;
    - mappings sum (an exchange may carry several metered calls);
    - ``model`` scales per §3.4 by FAMILY PREFIX (a dated snapshot id
      counts as its family, the #186 rule): ``claude-haiku*``/``None``
      ×1; ``claude-sonnet*`` ×SONNET_ENERGY_MULTIPLIER on every bound;
      ``claude-opus*`` ×OPUS_ENERGY_MULTIPLIER_LOW on the low bound,
      ×OPUS_ENERGY_MULTIPLIER_HIGH on the high, mid-point on the
      central — the doc flags these as extrapolated;
    - zero usage → :data:`WH_ZERO`; bounds always ordered low ≤ central
      ≤ high.
    """
    bounds: dict[str, float] = {"low": 0.0, "central": 0.0, "high": 0.0}
    for usage in usage_mappings:
        counts = usage_token_counts(usage)
        # §3: E_IN covers plain input AND cache-creation (write) tokens.
        input_class = counts["input_tokens"] + counts["cache_creation_input_tokens"]
        cache_read = counts["cache_read_input_tokens"]
        output = counts["output_tokens"]
        for end in _FACTOR_ENDS:
            e_in = getattr(E_IN_WH_PER_1K, end)
            e_out = getattr(E_OUT_WH_PER_1K, end)
            cache_read_factor = getattr(CACHE_READ_FACTOR, end)
            bounds[end] += (
                input_class / 1000 * e_in
                + cache_read / 1000 * e_in * cache_read_factor
                + output / 1000 * e_out
            )
    mult_low, mult_central, mult_high = _model_multipliers(model)
    return WhRange(
        low=bounds["low"] * mult_low,
        central=bounds["central"] * mult_central,
        high=bounds["high"] * mult_high,
    )


def local_energy_wh(cpu_seconds: float) -> WhRange:
    """The §5 measured-local slice: CPU-seconds → Wh range.

    RED-phase contract stub; pins: ``cpu_seconds × W_PER_VCPU ×
    PUE_HETZNER / 3600`` per bound. ``cpu_seconds`` is a genuine
    measurement (``resource.getrusage`` delta, instrumented at deploy);
    only the wattage conversion is estimated — the page labels the slice
    "measured CPU time × estimated per-vCPU wattage". Negative input
    raises ``ValueError`` (a measurement cannot be negative).
    """
    if cpu_seconds < 0:
        raise ValueError(f"cpu_seconds must be a non-negative measurement, got {cpu_seconds!r}")
    return WhRange(
        low=cpu_seconds * W_PER_VCPU.low * PUE_HETZNER.low / 3600,
        central=cpu_seconds * W_PER_VCPU.central * PUE_HETZNER.central / 3600,
        high=cpu_seconds * W_PER_VCPU.high * PUE_HETZNER.high / 3600,
    )


def co2e_grams(api_wh: WhRange, local_wh: WhRange = WH_ZERO) -> GramsCO2eRange:
    """The §2 CO2e term: gCO2e range from the two energy ranges.

    RED-phase contract stub; pins: per bound,
    ``api_wh × CIF_API/1000 + local_wh × CIF_LOCAL_G_PER_KWH/1000``
    (gCO2e per kWh applied to Wh) — location-based on BOTH supply
    chains, the §4 treatment.
    """
    return GramsCO2eRange(
        low=api_wh.low * CIF_API_G_PER_KWH.low / 1000 + local_wh.low * CIF_LOCAL_G_PER_KWH / 1000,
        central=api_wh.central * CIF_API_G_PER_KWH.central / 1000
        + local_wh.central * CIF_LOCAL_G_PER_KWH / 1000,
        high=api_wh.high * CIF_API_G_PER_KWH.high / 1000
        + local_wh.high * CIF_LOCAL_G_PER_KWH / 1000,
    )


def sum_wh_ranges(ranges: Iterable[WhRange]) -> WhRange:
    """Element-wise sum: session cumulative = sum of exchange ranges (§2).

    RED-phase contract stub; pins: bound-by-bound addition; the empty
    iterable sums to :data:`WH_ZERO`.
    """
    low = central = high = 0.0
    for wh_range in ranges:
        low += wh_range.low
        central += wh_range.central
        high += wh_range.high
    return WhRange(low=low, central=central, high=high)


# ---------------------------------------------------------------------------
# §8 everyday-equivalent anchors (each with a citable source)
# ---------------------------------------------------------------------------


def streaming_seconds_equivalent(co2e: GramsCO2eRange) -> tuple[float, float, float]:
    """Seconds of video streaming (IEA/Kamiya 2020: ≈36 gCO2e/hour).

    RED-phase contract stub; pins ``grams / 36 × 3600`` per bound
    (central exchange ≈ 0.2 g → ≈ 20 s, the doc's own anchor check).
    """

    def seconds(grams: float) -> float:
        return grams / STREAMING_G_CO2E_PER_HOUR * 3600

    return (seconds(co2e.low), seconds(co2e.central), seconds(co2e.high))


def metres_driven_equivalent(co2e: GramsCO2eRange) -> tuple[float, float, float]:
    """Metres driven by a typical passenger car (US EPA ≈ 0.25 g/metre).

    RED-phase contract stub; pins ``grams / 0.25`` per bound.
    """

    def metres(grams: float) -> float:
        return grams / CAR_G_CO2E_PER_METRE

    return (metres(co2e.low), metres(co2e.central), metres(co2e.high))


def exchanges_per_mug_of_tea(exchange_wh: WhRange) -> tuple[float, float, float]:
    """How many exchanges equal one mug of tea (31 Wh, first-principles).

    RED-phase contract stub; pins ``TEA_MUG_WH / exchange_wh`` with the
    bounds INVERTED honestly (the high-energy bound yields the LOW
    exchange count and vice versa), returned (low_count, central,
    high_count); a zero-energy bound raises ``ValueError`` rather than
    dividing by zero.
    """
    if exchange_wh.low <= 0 or exchange_wh.central <= 0 or exchange_wh.high <= 0:
        raise ValueError(
            "exchanges_per_mug_of_tea needs a strictly-positive energy bound on "
            f"every end, got {exchange_wh!r}"
        )
    # Honest inversion: the HIGH-energy bound gives the FEWEST exchanges.
    return (
        TEA_MUG_WH / exchange_wh.high,
        TEA_MUG_WH / exchange_wh.central,
        TEA_MUG_WH / exchange_wh.low,
    )


# ---------------------------------------------------------------------------
# Footer formatting (§9 — verbatim template, register rules enforced)
# ---------------------------------------------------------------------------


def format_wh_value(value: float) -> str:
    """One Wh figure for the footer: human-scale, never scientific.

    RED-phase contract stub; pins: at most two significant digits,
    trailing zeros stripped (0.1234→"0.12", 1.5→"1.5", 12.0→"12",
    0.004→"0.004"), never scientific notation, never a bare trailing
    dot.
    """
    if value == 0:
        return "0"
    # Round to two significant figures via Decimal (so 123.456 → "120",
    # 0.1234 → "0.12"), then render in plain notation — never scientific,
    # never a bare trailing dot — with trailing zeros stripped.
    decimal_value = Decimal(str(value))
    most_significant = decimal_value.adjusted()
    quantum = Decimal(1).scaleb(most_significant - 1)  # keep 2 sig figs
    rounded = decimal_value.quantize(quantum, rounding=ROUND_HALF_UP)
    rendered = format(rounded, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def format_wh_bound(value: float, *, end: str) -> str:
    """One RANGE BOUND for the footer, rounded OUTWARD (finding #364).

    RED-phase contract stub: raises ``NotImplementedError``; the failing
    suite in ``tests/unit/test_footprint_estimation.py``
    (``TestOutwardBoundFormatting``) pins the contract:

    - symmetric half-up rounding NARROWS a displayed range at both ends
      (a low bound can round UP, a high bound DOWN) — the §9 register
      contract says the displayed range is an honest propagation, so
      display rounding must be conservative: ``end="low"`` rounds DOWN
      (``ROUND_FLOOR``), ``end="high"`` rounds UP (``ROUND_CEILING``),
      and the rendered range always CONTAINS the computed one
      (``Decimal(rendered_low) <= value <= Decimal(rendered_high)``);
    - the :func:`format_wh_value` register rules are unchanged: at most
      two significant figures, plain notation (never scientific),
      trailing zeros stripped, never a bare trailing dot, zero → "0";
    - any ``end`` other than ``"low"``/``"high"`` raises ``ValueError``
      (never a silently mis-rounded bound);
    - :func:`format_footprint_footer` and
      :func:`format_footprint_footer_cached` render their bounds through
      this outward rule (central figures, if ever displayed, may stay
      half-up via :func:`format_wh_value`).
    """
    raise NotImplementedError(
        "format_wh_bound is a red-phase contract stub (review finding #364): "
        "outward bound rounding is pinned by TestOutwardBoundFormatting"
    )


def format_footprint_footer(answer_wh: WhRange, session_wh: WhRange) -> str:
    """The footer indicator line, VERBATIM per §9's template.

    RED-phase contract stub; pins: exactly
    :data:`FOOTPRINT_FOOTER_TEMPLATE` with the four figures rendered by
    :func:`format_wh_value`; ALWAYS a range (low and high both shown,
    from the propagated bounds); no gCO2e anywhere in the output.
    """
    return FOOTPRINT_FOOTER_TEMPLATE.format(
        lo=format_wh_value(answer_wh.low),
        hi=format_wh_value(answer_wh.high),
        session_lo=format_wh_value(session_wh.low),
        session_hi=format_wh_value(session_wh.high),
    )


def format_footprint_footer_cached(session_wh: WhRange) -> str:
    """The cached-replay indicator line (the flagged cached decision).

    RED-phase contract stub; pins: exactly
    :data:`FOOTPRINT_FOOTER_CACHED_TEMPLATE` with the session figures
    rendered by :func:`format_wh_value`.
    """
    return FOOTPRINT_FOOTER_CACHED_TEMPLATE.format(
        session_lo=format_wh_value(session_wh.low),
        session_hi=format_wh_value(session_wh.high),
    )


# ---------------------------------------------------------------------------
# The application-total ledger (#217 conventions, privacy-safe by schema)
# ---------------------------------------------------------------------------

#: The aggregate journal filename, written under the SAME ``state_dir``
#: as ``service.budget.SPEND_STATE_FILENAME`` — the footprint counter
#: sits beside the spend journal it mirrors (#217).
FOOTPRINT_STATE_FILENAME = "footprint-state.json"

#: THE privacy schema: the journal may contain these keys and NOTHING
#: else — token totals, an exchange count, measured CPU-seconds and the
#: since-date. No content, no identifiers, no per-exchange rows, nothing
#: joinable to a conversation. Pinned structurally against the file the
#: ledger actually writes.
FOOTPRINT_JOURNAL_ALLOWED_KEYS = frozenset(
    {
        "since",
        "exchanges",
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "cpu_seconds",
    }
)

#: The four provider-reported token counters the ledger accumulates (the
#: same four :func:`usage_token_counts` normalises).
_COUNT_KEYS: tuple[str, ...] = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


@dataclass(frozen=True)
class FootprintTotals:
    """The lifetime aggregate the /footprint page renders (§9.1)."""

    since: str | None
    exchanges: int
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    cpu_seconds: float


class FootprintLedgerError(Exception):
    """The aggregate journal is unreadable/corrupt: the totals are
    UNKNOWN — surfaced honestly, never replaced with silent zeros and
    never overwritten with a fresh count that erases history."""


class FootprintLedger:
    """Privacy-safe persistent aggregate: token totals + exchange count.

    RED-phase contract stub: construction records the seams; behaviour
    raises ``NotImplementedError``. The failing suite in
    ``tests/unit/test_footprint_ledger.py`` pins the #217 conventions:

    - ``record_exchange(usage_records, cpu_seconds=0.0)`` accepts the
      exchange log's ``usage_records`` shape (``{"model", "usage"}``
      mappings — the SAME records the spend cap charges) and adds their
      token counts to the lifetime totals, increments the exchange
      count by one, and adds ``cpu_seconds``; it journals atomically
      (``atomic_write_text``, fsync — finding #302) on EVERY record,
      under a lock, so a crash-loop never loses the aggregate;
      concurrent records lose nothing.
    - ``totals()`` returns the current :class:`FootprintTotals`; a fresh
      ledger (no journal yet) reads zero totals with ``since=None``.
    - A restart reads the journal back: totals continue, ``since`` (the
      first-ever record's UTC date, from the injected clock) is
      preserved — a redeploy can never reset the public count.
    - A corrupt/unreadable journal makes BOTH ``totals()`` and
      ``record_exchange`` raise :class:`FootprintLedgerError` naming the
      path: unknown totals are reported unknown (the page renders
      :data:`FOOTPRINT_TOTALS_UNAVAILABLE_NOTICE`), and history is never
      clobbered by restarting the count from zero.
    - **Privacy by schema:** the journal file's JSON object carries
      EXACTLY :data:`FOOTPRINT_JOURNAL_ALLOWED_KEYS`; no key from
      ``service.exchange_log.FORBIDDEN_IDENTIFIER_FIELDS``, no
      question/answer text, no exchange ids, no timestamps beyond the
      since-date — regardless of what extra keys ride in on the usage
      records.
    - ``state_path`` exposes the journal location so the composition
      root's "beside the spend state" wiring is pinnable.
    """

    def __init__(
        self,
        *,
        state_dir: Path,
        clock: Callable[[], datetime],
    ) -> None:
        self._state_dir = Path(state_dir)
        self._clock = clock
        self._lock = threading.Lock()

    @property
    def state_path(self) -> Path:
        """Where the aggregate journal lives (beside the spend journal)."""
        return self._state_dir / FOOTPRINT_STATE_FILENAME

    def _read_state(self) -> dict[str, Any] | None:
        """The journal as a dict, ``None`` when it has never been written.

        A present-but-corrupt journal raises :class:`FootprintLedgerError`
        naming the path — unknown totals are reported unknown, and history
        is never clobbered by treating a corrupt file as a fresh count.
        """
        path = self.state_path
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise FootprintLedgerError(
                f"footprint aggregate journal at {path} is unreadable or corrupt "
                f"({exc}) — the running totals are UNKNOWN, not zero"
            ) from exc
        if not isinstance(data, dict):
            raise FootprintLedgerError(
                f"footprint aggregate journal at {path} is not a JSON object — "
                "the running totals are UNKNOWN, not zero"
            )
        return data

    def record_exchange(
        self,
        usage_records: Sequence[Mapping[str, Any]],
        *,
        cpu_seconds: float = 0.0,
    ) -> None:
        """Add one exchange's token counts + CPU-seconds to the aggregate."""
        with self._lock:
            # Read the current aggregate FIRST: a corrupt journal raises here,
            # before any write, so history is never clobbered (BOTH read and
            # write refuse loudly on corruption).
            state = self._read_state()
            if state is None:
                state = {key: 0 for key in _COUNT_KEYS}
                state["since"] = None
                state["cpu_seconds"] = 0.0

            for record in usage_records:
                counts = usage_token_counts(record.get("usage") or {})
                for key in _COUNT_KEYS:
                    state[key] = int(state.get(key, 0)) + counts[key]
            state["exchanges"] = int(state.get("exchanges", 0)) + 1
            state["cpu_seconds"] = float(state.get("cpu_seconds", 0.0)) + float(cpu_seconds)
            if not state.get("since"):
                # The first-ever record's UTC date — preserved across restarts
                # so a redeploy can never reset the public since-date.
                state["since"] = self._clock().date().isoformat()

            # Privacy by schema: journal EXACTLY the allowed keys and nothing
            # else, whatever poison rode in on the usage records.
            clean = {key: state[key] for key in FOOTPRINT_JOURNAL_ALLOWED_KEYS}
            atomic_write_text(self.state_path, json.dumps(clean))

    def totals(self) -> FootprintTotals:
        """The lifetime aggregate (raises FootprintLedgerError when unknown)."""
        with self._lock:
            state = self._read_state()
        if state is None:
            return FootprintTotals(
                since=None,
                exchanges=0,
                input_tokens=0,
                output_tokens=0,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
                cpu_seconds=0.0,
            )
        return FootprintTotals(
            since=state.get("since"),
            exchanges=int(state.get("exchanges", 0)),
            input_tokens=int(state.get("input_tokens", 0)),
            output_tokens=int(state.get("output_tokens", 0)),
            cache_read_input_tokens=int(state.get("cache_read_input_tokens", 0)),
            cache_creation_input_tokens=int(state.get("cache_creation_input_tokens", 0)),
            cpu_seconds=float(state.get("cpu_seconds", 0.0)),
        )


# Referenced so the atomic-write dependency is explicit in the stub; the
# green implementation journals through it (the #217/#302 choreography).
_ATOMIC_WRITE = atomic_write_text
