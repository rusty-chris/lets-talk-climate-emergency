"""Footprint feature RED — the pure estimation model (service.footprint).

The owner approved docs/FOOTPRINT-METHODOLOGY.md (PR #355) as BINDING:
factors, ranges, footer wording verbatim, the always-a-range rule, no
gCO2e in the footer, the measured-vs-estimated split, the anchors. This
suite pins:

- **docs-as-code parity**: every §3 factor constant equals the doc's
  table (parsed from the markdown), and every verbatim presentation
  constant appears verbatim in the doc — the code can never drift from
  the published method;
- the §2 arithmetic with hand-computed vectors (the claude-carbon
  golden-test-vector pattern the doc adopts), including the doc's own
  "typical exchange" check;
- honest range propagation: low bounds with low factor ends, high with
  high — never a point estimate;
- the §5 getrusage-CPU-seconds → measured-local-Wh conversion;
- the §8 anchor arithmetic against the doc's stated equivalents;
- the §9 footer template rendered VERBATIM, with the register rules
  (always a range; "est."/"estimates" present; the /footprint link;
  NO gCO2e; Wh not J/kWh) enforced structurally;
- the $0 rule structurally: the module imports no provider SDK, HTTP
  client or web framework — all arithmetic is local.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

import service.footprint as footprint
from service.footprint import (
    ANTHROPIC_GRID_ASSUMPTION_SENTENCE,
    ANTHROPIC_UNCERTAINTY_PARAGRAPH,
    CACHE_READ_FACTOR,
    CAR_G_CO2E_PER_METRE,
    CIF_API_G_PER_KWH,
    CIF_LOCAL_DE_G_PER_KWH,
    CIF_LOCAL_FI_G_PER_KWH,
    CIF_LOCAL_G_PER_KWH,
    E_IN_WH_PER_1K,
    E_OUT_WH_PER_1K,
    FOOTPRINT_ESTIMATES_PHRASE,
    FOOTPRINT_FOOTER_CACHED_TEMPLATE,
    FOOTPRINT_FOOTER_TEMPLATE,
    FOOTPRINT_ROUTE,
    MARKET_VS_LOCATION_SENTENCE,
    OPUS_ENERGY_MULTIPLIER_HIGH,
    OPUS_ENERGY_MULTIPLIER_LOW,
    PUE_HETZNER,
    SONNET_ENERGY_MULTIPLIER,
    STREAMING_G_CO2E_PER_HOUR,
    TEA_MUG_WH,
    TRAINING_EXCLUSION_PHRASE,
    W_PER_VCPU,
    WH_ZERO,
    GramsCO2eRange,
    WhRange,
    api_energy_wh,
    co2e_grams,
    exchanges_per_mug_of_tea,
    format_footprint_footer,
    format_footprint_footer_cached,
    format_wh_value,
    local_energy_wh,
    metres_driven_equivalent,
    streaming_seconds_equivalent,
    sum_wh_ranges,
    usage_token_counts,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
METHODOLOGY_DOC = REPO_ROOT / "docs" / "FOOTPRINT-METHODOLOGY.md"


def doc_text() -> str:
    return METHODOLOGY_DOC.read_text(encoding="utf-8")


def normalised_doc() -> str:
    """The doc with markdown emphasis stripped and whitespace collapsed,
    so verbatim constants can be matched across hard-wrapped lines."""
    return " ".join(doc_text().replace("**", "").replace("*", "").replace("> ", " ").split())


def doc_factor_row(name: str) -> tuple[float, tuple[float, float] | None]:
    """Parse one §3 table row: (central, (range_low, range_high) | None)."""
    for line in doc_text().splitlines():
        if line.strip().startswith(f"| `{name}`"):
            cells = [cell.strip().replace("**", "") for cell in line.split("|")]
            # cells[0] is empty (leading pipe); [1] name, [2] central, [3] range
            central = float(cells[2])
            range_cell = cells[3]
            if range_cell in ("", "—", "-"):
                return central, None
            low_text, high_text = re.split(r"\s*–\s*", range_cell)
            return central, (float(low_text), float(high_text))
    raise AssertionError(f"§3 table row for {name} not found in {METHODOLOGY_DOC}")


class TestFactorsMatchTheMethodologyDoc:
    """Docs-as-code: the constants ARE the published table (BINDING)."""

    @pytest.mark.parametrize(
        ("row_name", "factor"),
        [
            ("E_OUT", E_OUT_WH_PER_1K),
            ("E_IN", E_IN_WH_PER_1K),
            ("CACHE_READ_FACTOR", CACHE_READ_FACTOR),
            ("CIF_API", CIF_API_G_PER_KWH),
            ("W_VCPU", W_PER_VCPU),
            ("PUE_HETZNER", PUE_HETZNER),
        ],
    )
    def test_factor_equals_the_doc_table(self, row_name: str, factor) -> None:
        central, bounds = doc_factor_row(row_name)
        assert factor.central == pytest.approx(central), f"{row_name} central drifted from the doc"
        assert bounds is not None, f"{row_name} should carry a range in the doc"
        assert factor.low == pytest.approx(bounds[0]), f"{row_name} low bound drifted"
        assert factor.high == pytest.approx(bounds[1]), f"{row_name} high bound drifted"

    def test_local_grid_factors_match_the_doc(self) -> None:
        # The CIF_LOCAL row is special-cased in the doc: "363 (DE) / ~95 (FI)".
        text = doc_text()
        assert "363 (DE)" in text and "95 (FI)" in text
        assert CIF_LOCAL_DE_G_PER_KWH == 363.0
        assert CIF_LOCAL_FI_G_PER_KWH == 95.0
        # The chosen default is the German (higher, location-based) factor —
        # this product does not buy its way out of a range (§4). FLAGGED
        # decision: revisit if the production DC is hel1 (Finland).
        assert CIF_LOCAL_G_PER_KWH == CIF_LOCAL_DE_G_PER_KWH

    def test_model_multipliers_match_the_doc(self) -> None:
        # §3.4: Opus ×3–4, Sonnet ×2 — flagged in the doc as the weakest
        # numbers in the table.
        text = normalised_doc()
        assert "Opus best-mode: ×3–4 on both token factors" in text
        assert "Sonnet bake-off arm: ×2" in text
        assert OPUS_ENERGY_MULTIPLIER_LOW == 3.0
        assert OPUS_ENERGY_MULTIPLIER_HIGH == 4.0
        assert SONNET_ENERGY_MULTIPLIER == 2.0

    def test_anchor_factors_match_the_doc_sources(self) -> None:
        # §8: IEA/Kamiya ≈36 g/hour; EPA ≈0.25 g/metre; 0.031 kWh per mug.
        text = doc_text()
        assert "36 gCO2e" in text
        assert "0.25 g/metre" in text
        assert "0.031 kWh" in text
        assert STREAMING_G_CO2E_PER_HOUR == 36.0
        assert CAR_G_CO2E_PER_METRE == 0.25
        assert TEA_MUG_WH == pytest.approx(31.0)

    @pytest.mark.parametrize(
        "constant",
        [
            FOOTPRINT_FOOTER_TEMPLATE,
            ANTHROPIC_UNCERTAINTY_PARAGRAPH,
            MARKET_VS_LOCATION_SENTENCE,
            ANTHROPIC_GRID_ASSUMPTION_SENTENCE,
            TRAINING_EXCLUSION_PHRASE,
        ],
        ids=[
            "footer-template",
            "anthropic-uncertainty",
            "market-vs-location",
            "grid-assumption",
            "training-exclusion",
        ],
    )
    def test_verbatim_constant_appears_in_the_doc(self, constant: str) -> None:
        normalised_constant = " ".join(constant.split())
        assert normalised_constant in normalised_doc(), (
            "verbatim constant drifted from docs/FOOTPRINT-METHODOLOGY.md:\n"
            f"{normalised_constant!r}"
        )

    def test_factors_version_is_a_dated_event(self) -> None:
        # §9.8: a factor update is a visible, dated event.
        assert re.match(r"^\d{4}-\d{2}", footprint.FOOTPRINT_FACTORS_VERSION)

    def test_footprint_route_constant(self) -> None:
        assert FOOTPRINT_ROUTE == "/footprint"


class TestUsageTokenCounts:
    def test_normalises_the_four_seam_keys(self) -> None:
        counts = usage_token_counts(
            {
                "input_tokens": 900,
                "output_tokens": 42,
                "cache_read_input_tokens": 4200,
                "cache_creation_input_tokens": 7,
            }
        )
        assert counts == {
            "input_tokens": 900,
            "output_tokens": 42,
            "cache_read_input_tokens": 4200,
            "cache_creation_input_tokens": 7,
        }

    def test_none_and_absent_keys_count_zero(self) -> None:
        # The service.budget convention: None/absent are zero, never a crash.
        counts = usage_token_counts({"input_tokens": None, "output_tokens": 10})
        assert counts["input_tokens"] == 0
        assert counts["output_tokens"] == 10
        assert counts["cache_read_input_tokens"] == 0
        assert counts["cache_creation_input_tokens"] == 0

    def test_unknown_keys_are_ignored(self) -> None:
        counts = usage_token_counts({"output_tokens": 5, "service_tier": "standard"})
        assert set(counts) == {
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        }


class TestApiEnergyModel:
    """The §2 E_api term — hand-computed vectors, honest propagation."""

    def test_input_tokens_use_e_in(self) -> None:
        result = api_energy_wh([{"input_tokens": 1000}])
        assert result.low == pytest.approx(0.005)
        assert result.central == pytest.approx(0.02)
        assert result.high == pytest.approx(0.07)

    def test_output_tokens_use_e_out(self) -> None:
        result = api_energy_wh([{"output_tokens": 1000}])
        assert result.low == pytest.approx(0.1)
        assert result.central == pytest.approx(0.5)
        assert result.high == pytest.approx(1.5)

    def test_cache_creation_charges_as_input(self) -> None:
        # §3: E_IN includes cache writes.
        result = api_energy_wh([{"cache_creation_input_tokens": 1000}])
        assert result.central == pytest.approx(0.02)

    def test_cache_read_uses_the_energy_factor_not_the_billing_ratio(self) -> None:
        # §3: 0.08 (0.05–0.20) — explicitly NOT Anthropic's 0.1× price.
        result = api_energy_wh([{"cache_read_input_tokens": 1000}])
        assert result.low == pytest.approx(0.005 * 0.05)
        assert result.central == pytest.approx(0.02 * 0.08)
        assert result.high == pytest.approx(0.07 * 0.20)

    def test_mappings_sum_over_an_exchange(self) -> None:
        # An exchange may carry several metered calls (§2: summed over
        # all usage_records).
        result = api_energy_wh([{"output_tokens": 500}, {"output_tokens": 500}])
        assert result.central == pytest.approx(0.5)

    def test_zero_usage_is_the_zero_range(self) -> None:
        assert api_energy_wh([]) == WH_ZERO
        assert api_energy_wh([{}]) == WH_ZERO

    def test_bounds_propagate_like_ends_together(self) -> None:
        """Low bound = every factor's low end; high = every high end —
        the doc's honest-propagation rule, never a mixed point."""
        usage = {"input_tokens": 7000, "output_tokens": 700}
        result = api_energy_wh([usage])
        assert result.low == pytest.approx(7 * 0.005 + 0.7 * 0.1)
        assert result.high == pytest.approx(7 * 0.07 + 0.7 * 1.5)
        assert result.low < result.central < result.high

    def test_the_docs_typical_exchange_golden_vector(self) -> None:
        """§3's own check: ≈7k input-class + ≈0.7k output tokens →
        E_api ≈ 0.1–1.5 Wh, central ≈ 0.5 Wh (the golden-test-vector
        pattern adopted from claude-carbon)."""
        result = api_energy_wh([{"input_tokens": 7000, "output_tokens": 700}])
        assert result.low == pytest.approx(0.105)
        assert result.central == pytest.approx(0.49)
        assert result.high == pytest.approx(1.54)

    def test_haiku_family_and_none_scale_by_one(self) -> None:
        base = api_energy_wh([{"output_tokens": 1000}])
        assert api_energy_wh([{"output_tokens": 1000}], model="claude-haiku-4-5") == base
        # A dated snapshot counts as its family (the #186 prefix rule).
        assert api_energy_wh([{"output_tokens": 1000}], model="claude-haiku-4-5-20260101") == base

    def test_sonnet_scales_every_bound_by_two(self) -> None:
        base = api_energy_wh([{"output_tokens": 1000}])
        scaled = api_energy_wh([{"output_tokens": 1000}], model="claude-sonnet-4-5")
        assert scaled.low == pytest.approx(base.low * 2)
        assert scaled.central == pytest.approx(base.central * 2)
        assert scaled.high == pytest.approx(base.high * 2)

    def test_opus_scales_low_x3_high_x4(self) -> None:
        # §3.4: ×3–4, applied as an honest range widening — low bound ×3,
        # high bound ×4, central at the mid-multiplier.
        base = api_energy_wh([{"output_tokens": 1000}])
        scaled = api_energy_wh([{"output_tokens": 1000}], model="claude-opus-4-6")
        assert scaled.low == pytest.approx(base.low * 3)
        assert scaled.high == pytest.approx(base.high * 4)
        assert scaled.central == pytest.approx(base.central * 3.5)

    def test_unknown_model_family_refuses_loudly(self) -> None:
        # Mirrors the pricing seam's unknown-model refusal: never a
        # silently wrong factor.
        with pytest.raises(ValueError):
            api_energy_wh([{"output_tokens": 1000}], model="gpt-fictional-9")


class TestLocalEnergy:
    """§5: measured CPU-seconds × estimated wattage × PUE — the one
    slice with a measured energy basis."""

    def test_one_vcpu_hour(self) -> None:
        result = local_energy_wh(3600.0)
        assert result.low == pytest.approx(0.7 * 1.1)
        assert result.central == pytest.approx(2.2 * 1.13)
        assert result.high == pytest.approx(3.8 * 1.2)

    def test_a_few_cpu_seconds_lands_in_the_docs_expected_order(self) -> None:
        # §3: "a few CPU-seconds → ≈ 0.002 – 0.01 Wh".
        result = local_energy_wh(3.0)
        assert result.central == pytest.approx(3.0 * 2.2 * 1.13 / 3600)
        assert 0.002 <= result.central <= 0.01

    def test_zero_is_zero(self) -> None:
        assert local_energy_wh(0.0) == WH_ZERO

    def test_negative_measurement_refuses(self) -> None:
        with pytest.raises(ValueError):
            local_energy_wh(-0.1)


class TestCO2e:
    def test_api_energy_times_grid_intensity(self) -> None:
        grams = co2e_grams(WhRange(low=0.105, central=0.49, high=1.54))
        assert grams.low == pytest.approx(0.105 * 287 / 1000)
        assert grams.central == pytest.approx(0.49 * 384 / 1000)
        assert grams.high == pytest.approx(1.54 * 450 / 1000)

    def test_local_slice_uses_the_location_based_local_grid(self) -> None:
        grams = co2e_grams(WH_ZERO, WhRange(low=1.0, central=1.0, high=1.0))
        assert grams.central == pytest.approx(CIF_LOCAL_G_PER_KWH / 1000)

    def test_typical_exchange_is_about_a_fifth_of_a_gram(self) -> None:
        # §3: CO2e ≈ 0.04 – 0.6 g (central ≈ 0.2 g).
        grams = co2e_grams(api_energy_wh([{"input_tokens": 7000, "output_tokens": 700}]))
        assert grams.central == pytest.approx(0.19, abs=0.03)
        assert grams.low < 0.05
        assert grams.high < 0.75


class TestSumRanges:
    def test_elementwise_sum(self) -> None:
        total = sum_wh_ranges(
            [
                WhRange(low=0.1, central=0.5, high=1.5),
                WhRange(low=0.2, central=0.3, high=0.4),
            ]
        )
        assert total == WhRange(low=pytest.approx(0.3), central=pytest.approx(0.8), high=1.9)

    def test_empty_sum_is_zero(self) -> None:
        assert sum_wh_ranges([]) == WH_ZERO


class TestAnchors:
    """§8 — each equivalent reproduces the doc's stated numbers."""

    def test_streaming_seconds(self) -> None:
        low, central, high = streaming_seconds_equivalent(
            GramsCO2eRange(low=0.04, central=0.2, high=0.6)
        )
        # The doc: central ≈ 0.2 g → ≈ 20 s; range ≈ 4–60 s.
        assert central == pytest.approx(20.0)
        assert low == pytest.approx(4.0)
        assert high == pytest.approx(60.0)

    def test_metres_driven(self) -> None:
        low, central, high = metres_driven_equivalent(
            GramsCO2eRange(low=0.04, central=0.2, high=0.6)
        )
        # The doc: "driving about one metre".
        assert central == pytest.approx(0.8)
        assert low == pytest.approx(0.16)
        assert high == pytest.approx(2.4)

    def test_exchanges_per_mug_of_tea(self) -> None:
        low_count, central, high_count = exchanges_per_mug_of_tea(
            WhRange(low=0.1, central=0.5, high=1.5)
        )
        # The doc: "About 60 exchanges ≈ one mug of tea (range 20–300)".
        assert central == pytest.approx(62.0)
        # Bounds invert honestly: the HIGH-energy bound gives the LOW count.
        assert low_count == pytest.approx(31 / 1.5)
        assert high_count == pytest.approx(310.0)
        assert low_count < central < high_count

    def test_tea_anchor_refuses_a_zero_energy_bound(self) -> None:
        with pytest.raises(ValueError):
            exchanges_per_mug_of_tea(WhRange(low=0.0, central=0.5, high=1.5))


class TestFooterFormatting:
    """§9: the exact wording, and the register rules as structure."""

    def test_wh_values_format_human_scale(self) -> None:
        assert format_wh_value(0.1234) == "0.12"
        assert format_wh_value(1.5) == "1.5"
        assert format_wh_value(12.0) == "12"
        assert format_wh_value(0.004) == "0.004"
        assert format_wh_value(123.456) == "120"

    def test_wh_values_never_use_scientific_notation(self) -> None:
        rendered = format_wh_value(1.23e-05)
        assert "e" not in rendered.lower()
        assert not rendered.endswith(".")

    def test_footer_is_the_template_verbatim(self) -> None:
        line = format_footprint_footer(
            WhRange(low=0.1, central=0.5, high=1.5),
            WhRange(low=0.4, central=2.0, high=6.0),
        )
        assert line == FOOTPRINT_FOOTER_TEMPLATE.format(
            lo="0.1", hi="1.5", session_lo="0.4", session_hi="6"
        )

    def test_footer_register_rules(self) -> None:
        line = format_footprint_footer(
            WhRange(low=0.1, central=0.5, high=1.5),
            WhRange(low=0.4, central=2.0, high=6.0),
        )
        # Always a RANGE: both propagated bounds, an en-dash between them.
        assert "0.1–1.5" in line
        # "est." is present and adjacent to the figures (it prefixes them).
        assert "est. 0.1–1.5 Wh" in line
        assert FOOTPRINT_ESTIMATES_PHRASE in line
        # The link out to the method.
        assert FOOTPRINT_ROUTE in line
        # NO gCO2e in the footer — the carbon figure belongs on the page
        # beside its disclosed grid assumption (§9).
        assert "co2" not in line.lower()
        assert "gram" not in line.lower()
        # Wh, not J or kWh.
        assert "kWh" not in line
        assert " J" not in line

    def test_cached_footer_is_the_cached_template_verbatim(self) -> None:
        line = format_footprint_footer_cached(WhRange(low=0.4, central=2.0, high=6.0))
        assert line == FOOTPRINT_FOOTER_CACHED_TEMPLATE.format(session_lo="0.4", session_hi="6")

    def test_cached_footer_register_rules(self) -> None:
        line = format_footprint_footer_cached(WhRange(low=0.4, central=2.0, high=6.0))
        assert "~0 Wh" in line
        assert "cached" in line
        assert FOOTPRINT_ESTIMATES_PHRASE in line
        assert FOOTPRINT_ROUTE in line
        assert "co2" not in line.lower()


class TestZeroApiCallsStructurally:
    """The $0 rule as structure: the estimation module is pure local
    arithmetic — no provider SDK, no HTTP client, no web framework."""

    FORBIDDEN_ROOTS = {
        "anthropic",
        "httpx",
        "requests",
        "aiohttp",
        "urllib",
        "socket",
        "fastapi",
        "starlette",
        "streamlit",
        "rag",
        "evals",
    }

    def test_module_imports_stay_pure(self) -> None:
        source = (REPO_ROOT / "service" / "footprint.py").read_text(encoding="utf-8")
        roots: set[str] = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.add(node.module.split(".")[0])
        offenders = roots & self.FORBIDDEN_ROOTS
        assert not offenders, (
            f"service/footprint.py imports {sorted(offenders)} — the footprint "
            "feature must be $0: pure local arithmetic, zero API calls"
        )

    def test_no_network_strings_in_the_module(self) -> None:
        source = (REPO_ROOT / "service" / "footprint.py").read_text(encoding="utf-8")
        assert "api.anthropic.com" not in source
        assert "http://" not in source and "https://" not in source
