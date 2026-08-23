"""Review finding #51 — RED: the SYNTHETIC FIXTURE marker's claim must be true.

The spike data fixtures' first lines claim "all values invented", but the
review verified several rows verbatim against the real archives:
Bereiter age/CO2 pairs (-8.56/316.33, -51.03/368.02), Mauna Loa annual
means (1959: 315.98, 1960: 316.91, 2024: 424.61) and the real GISTEMP
J-D values on the 1880 (-.20) and 2024 (1.28) rows. A marker that reads
"invented" over copied rows is decoration: a reviewer who trusts it will
not re-check, which is how a licence-encumbered Tier B/C row eventually
ships (IMPLEMENTATION §5 / DESIGN §2.1).

Two enforceable guards, following the posture the #14 fixtures under
tests/fixtures/datasets/raw/ already adopted:

1. Every synthetic data fixture that mimics a named real source must
   STATE its row-copy posture in the header — either the no-copy
   declaration ("No rows are copied from the real file", the
   datasets/raw convention citing this finding) or an explicit
   "rows ... copied from ... under <licence>" declaration naming what
   was copied and on what terms.
2. The rows the review verified as verbatim-real must be absent from the
   spike fixtures (the no-copy posture made true, not re-worded).

Constraint for the implementer (recorded here, enforced by the dependent
suites): perturbed replacements must keep the structural properties the
parser/transform/render tests exercise — BOM+CRLF and one negative
(post-1950) age in the Bereiter file, the '***' in-progress-year row in
the GISTEMP file, shuffled row order, realistic magnitudes — and
tests/unit/test_spike_parsers.py now derives its expected values from
the fixture text instead of hard-coding them, so regeneration does not
require editing that suite.
"""

from __future__ import annotations

import re
from pathlib import Path

FIXTURES_ROOT = Path(__file__).resolve().parents[1] / "fixtures"
SPIKE = FIXTURES_ROOT / "spike"
DATASETS_RAW = FIXTURES_ROOT / "datasets" / "raw"

#: Tabular data suffixes — the mimic-a-real-source fixture population.
#: (synthetic_doc.py is a chunker text fixture, not tabular data.)
DATA_SUFFIXES = {".csv", ".tsv", ".txt"}

#: The no-copy posture declaration (the tests/fixtures/datasets/raw/
#: convention: "No rows are copied from the real file (review finding #51)").
NO_COPY_RE = re.compile(r"(?i)no rows (?:are|were) copied")

#: The explicit-copy posture: name the copied rows and the licence terms.
EXPLICIT_COPY_RE = re.compile(
    r"(?i)rows?\b[^\n]*\bcopied from\b[^\n]*(public.domain|licen[cs]e|cc.by)"
)

#: (file, banned token, why) — values the #51 review verified verbatim
#: against the real archives. The token must not appear anywhere in the
#: file: the fixture must be regenerated with perturbed values.
KNOWN_VERBATIM_TOKENS = [
    ("spike/bereiter_synthetic.txt", "-8.56", "real Bereiter age of the 316.33 ppm sample"),
    ("spike/bereiter_synthetic.txt", "316.33", "real Bereiter CO2 value (age -8.56)"),
    ("spike/bereiter_synthetic.txt", "-51.03", "real Bereiter age of the 368.02 ppm sample"),
    ("spike/bereiter_synthetic.txt", "368.02", "real Bereiter CO2 value (age -51.03)"),
    ("spike/gml_synthetic.csv", "315.98", "real Mauna Loa 1959 annual mean"),
    ("spike/gml_synthetic.csv", "316.91", "real Mauna Loa 1960 annual mean"),
    ("spike/gml_synthetic.csv", "424.61", "real Mauna Loa 2024 annual mean"),
]

#: (year, real J-D value) — the GISTEMP rows the review verified carry
#: the real annual means; checked cell-precisely because short anomaly
#: strings would over-match as bare tokens.
KNOWN_VERBATIM_GISTEMP_JD = [(1880, "-.20"), (2024, "1.28")]


def _data_fixture_files() -> list[Path]:
    files = [
        path
        for directory in (SPIKE, DATASETS_RAW)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix in DATA_SUFFIXES
    ]
    assert files, "expected mimicking data fixtures under tests/fixtures/"
    return files


def _header_text(path: Path, count: int = 12) -> str:
    """The file's header comment block as one whitespace-normalised line
    (posture sentences wrap across '#'-prefixed comment lines)."""
    lines = path.read_text(encoding="utf-8-sig").splitlines()[:count]
    return " ".join(line.lstrip("#").strip() for line in lines)


def test_mimicking_fixtures_state_their_row_copy_posture():
    """Guard 1: the marker's guarantee is stated per-file in an
    enforceable form. Every tabular fixture mimicking a real source must
    declare, in its header, either that no rows are copied from the real
    file or exactly which rows are copied and under what licence — the
    convention the #14 fixtures already follow. RED for the four spike
    fixtures (and the owid sample, which states invention but not the
    canonical no-copy declaration)."""
    missing = [
        str(path.relative_to(FIXTURES_ROOT))
        for path in _data_fixture_files()
        if not (
            NO_COPY_RE.search(_header_text(path)) or EXPLICIT_COPY_RE.search(_header_text(path))
        )
    ]
    assert missing == [], (
        "these mimicking data fixtures state no row-copy posture in their "
        f"header: {missing} — add the 'No rows are copied from the real file "
        "(review finding #51)' declaration (after making it true), or declare "
        "exactly which rows are copied and under what licence"
    )


def test_spike_fixtures_carry_no_known_verbatim_rows():
    """Guard 2: the values the #51 review verified against the real
    archives are gone — the no-copy claim made true by perturbation.
    RED while the verbatim Bereiter/Mauna-Loa rows remain."""
    offending = []
    for relative, token, why in KNOWN_VERBATIM_TOKENS:
        path = FIXTURES_ROOT / relative
        assert path.is_file(), f"{relative}: fixture missing"
        if token in path.read_text(encoding="utf-8-sig"):
            offending.append(f"{relative}: {token} ({why})")
    assert offending == [], (
        "verbatim real-archive values found in fixtures claiming 'all values "
        f"invented': {offending} — perturb/regenerate these rows (keeping the "
        "structural properties the parser tests exercise)"
    )


def test_gistemp_fixture_jd_values_are_not_the_real_annual_means():
    """Guard 2, GISTEMP: the 1880 and 2024 rows must not carry the real
    J-D annual means the review verified (-.20 and 1.28). Checked at the
    J-D cell (not as bare substrings: short anomaly strings recur
    legitimately across invented monthly columns)."""
    path = SPIKE / "gistemp_synthetic.csv"
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    header = lines[1].split(",")
    year_column = header.index("Year")
    jd_column = header.index("J-D")
    jd_by_year = {}
    for line in lines[2:]:
        cells = line.split(",")
        jd_by_year[int(cells[year_column])] = cells[jd_column]

    offending = [
        f"{year}: J-D {real_value!r} is the real GISTEMP annual mean"
        for year, real_value in KNOWN_VERBATIM_GISTEMP_JD
        if jd_by_year.get(year) == real_value
    ]
    assert offending == [], (
        f"gistemp_synthetic.csv carries verbatim real J-D values: {offending} "
        "— perturb these rows (the file claims 'all values invented')"
    )
