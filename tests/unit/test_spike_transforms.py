"""Characterisation tests pinning the spike's BP->CE, rebaseline and splice
arithmetic on tiny inline synthetic frames.

These pin the three transforms the flagship chart requires (DESIGN 3.7,
found missing from the original vocabulary in review). The hand-computed
values here match the spot-checks in reviews/spike-04-chart-findings.md and
seed the independent gold fixtures of #17/#20.
"""

import pandas as pd
import pytest

from charts.spike import transforms


def frame(**cols) -> pd.DataFrame:
    return pd.DataFrame(cols)


class TestBpToCe:
    def test_endpoints(self):
        # SYNTHETIC FIXTURE (inline): the flagship's real endpoints, hand-computed.
        df = frame(age_bp=[0.0, 100.0, 12000.0, -51.03], v=[1.0, 2.0, 3.0, 4.0])
        out = transforms.bp_to_ce(df)
        assert list(out.columns) == ["year_ce", "v"]
        got = dict(zip(out["v"], out["year_ce"], strict=True))
        assert got[1.0] == 1950.0  # 0 BP = 1950 CE by convention
        assert got[2.0] == 1850.0  # Kaufman reference bin centre
        assert got[3.0] == -10050.0  # 12000 BP
        assert got[4.0] == pytest.approx(2001.03)  # negative BP = post-1950

    def test_output_sorted_time_ascending(self):
        out = transforms.bp_to_ce(frame(age_bp=[0.0, 12000.0, 100.0], v=[1, 2, 3]))
        assert out["year_ce"].is_monotonic_increasing


class TestRebaseline:
    def test_period_mean_becomes_zero_and_shift_is_uniform(self):
        # SYNTHETIC FIXTURE (inline): 1880-1900 mean is (-0.2 - 0.3 - 0.1)/3 = -0.2.
        df = frame(year_ce=[1880.0, 1890.0, 1900.0, 2024.0], t=[-0.2, -0.3, -0.1, 1.28])
        out = transforms.rebaseline(df, "t", (1880, 1900))
        window = out[(out.year_ce >= 1880) & (out.year_ce <= 1900)]["t"]
        assert window.mean() == pytest.approx(0.0)
        assert out[out.year_ce == 2024.0].iloc[0]["t"] == pytest.approx(1.48)

    def test_input_frame_not_mutated(self):
        df = frame(year_ce=[1880.0, 2000.0], t=[-0.2, 1.0])
        transforms.rebaseline(df, "t", (1880, 1880))
        assert df["t"].tolist() == [-0.2, 1.0]

    def test_empty_alignment_period_rejected(self):
        df = frame(year_ce=[2000.0], t=[1.0])
        with pytest.raises(ValueError, match="contains no data"):
            transforms.rebaseline(df, "t", (1880, 1900))


class TestSplice:
    def paleo(self):
        return frame(year_ce=[1750.0, 1850.0, 1958.56], v=[0.05, 0.0, 316.33], extra=[1, 2, 3])

    def instrumental(self):
        return frame(year_ce=[1959.0, 1960.0], v=[315.98, 316.91])

    def test_boundary_is_strict_below_then_inclusive_above(self):
        out = transforms.splice(self.paleo(), self.instrumental(), 1959.0)
        paleo_years = out[out.segment == "paleo"]["year_ce"]
        instr_years = out[out.segment == "instrumental"]["year_ce"]
        assert paleo_years.max() < 1959.0
        assert instr_years.min() == 1959.0
        assert len(out) == 5

    def test_segment_labels_and_common_columns_only(self):
        out = transforms.splice(self.paleo(), self.instrumental(), 1959.0)
        assert list(out.columns) == ["year_ce", "v", "segment"]  # 'extra' dropped
        assert set(out["segment"]) == {"paleo", "instrumental"}
        assert out["year_ce"].is_monotonic_increasing

    def test_one_sided_splice_rejected(self):
        # A splice year that leaves one side empty would render without a
        # visible join, making the mandatory splice annotation a lie.
        with pytest.raises(ValueError, match="must be non-empty"):
            transforms.splice(self.paleo(), self.instrumental(), 1700.0)
        with pytest.raises(ValueError, match="must be non-empty"):
            transforms.splice(self.paleo(), self.instrumental(), 2050.0)

    def test_missing_year_column_rejected(self):
        with pytest.raises(ValueError, match="year_ce"):
            transforms.splice(frame(a=[1]), frame(b=[2]), 1959.0)
