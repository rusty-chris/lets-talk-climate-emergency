"""Corpus/dataset manifest schema + licensing invariants (issue #5).

Pure over mappings and paths passed in (IMPLEMENTATION.md §1) — no
filesystem reach-around, no network. Implements the §2.1 invariants named
in DESIGN.md, as amended by ADR-023 (no dataset files in git + sha256
verification of fetched files) and review findings #45/#46 (a licence
claim requires evidence on file; a multi-origin dataset carries per-segment
provenance and every segment's credit must appear in the rendered
attribution).

Conventions:

- Refusals raise :class:`ManifestError`. A refusal message names the
  offending document/dataset/pair id and every violated field — all
  violations found for an entry are aggregated into one message, not just
  the first.
- Loaded records expose the manifest fields as attributes (dataclasses),
  typed: bools are bools, dates are ``datetime.date``, ``human_signoff``
  is a record with ``who``/``date``/``note`` attributes, provenance
  segments are records with ``origin``/``period``/``licence``/
  ``licence_evidence``/``credit`` attributes.
- Loaders tolerate unknown top-level keys (``version``, ``access_date``,
  and the fixtures-only ``violations`` section are not entries).
"""

from __future__ import annotations

import datetime
import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

#: Values a *corpus document* may carry (DESIGN.md §2.1 — exactly three).
DOCUMENT_PERMITTED_CONTEXTS = frozenset(
    {"open", "non-commercial-educational", "permission-on-file"}
)

#: Values a *dataset* may carry. Datasets additionally admit
#: ``open-provisional`` (ADR-023 / review #45): an unconfirmed open verdict
#: that requires a ``licence_note`` recording the evidence trail and is
#: never allowed in the chart data pack.
DATASET_PERMITTED_CONTEXTS = frozenset(DOCUMENT_PERMITTED_CONTEXTS | {"open-provisional"})

#: First-line marker that exempts a committed data-like file from the
#: ADR-023 no-dataset-files-in-git check (same marker the #24 fixture
#: corpus uses; match on this substring, tolerating a BOM and either dash).
SYNTHETIC_FIXTURE_MARKER = "SYNTHETIC FIXTURE"

#: Extensions treated as "data-like" for the ADR-023 no-committed-data check.
_DATA_LIKE_SUFFIXES = frozenset({".csv", ".tsv", ".txt", ".nc", ".parquet"})

#: Operational-telemetry files that ADR-023 does NOT govern but whose extension
#: matches ``_DATA_LIKE_SUFFIXES``. ADR-023's decision is scoped to *real dataset
#: files* (NOAA/NASA/... climate data) and corpus text — not operational
#: metadata. The spend ledger (dev-cost-plan M8, `reviews/dev-cost-plan-2026-08.md`)
#: is mandated to be committed, carries only API token/cost counts (no licensed
#: data), and would otherwise trip the suffix heuristic. Allow-listed by exact
#: repo-relative path so nothing dataset-shaped can slip through.
_ADR023_OPERATIONAL_ALLOWLIST = frozenset({"evals/spend-ledger.csv"})


class ManifestError(ValueError):
    """A licensing-invariant violation. The message names the offending

    document/dataset/pair id and every violated field (issue #5 acceptance
    criterion: refusal messages name the offending document and field).
    """


# ---------------------------------------------------------------------------
# Typed records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HumanSignoff:
    who: str
    date: datetime.date | None
    note: str


@dataclass(frozen=True)
class DocumentRecord:
    id: str
    licence: str
    licence_evidence: str
    attribution_text: str
    canonical_url: str
    redistributable: bool
    permitted_context: str
    permission_evidence: Any
    consensus_position: str
    sha256: str
    retrieved_at: datetime.date | None
    source_tier: str
    human_signoff: HumanSignoff
    path: str | None = None
    source_url: str | None = None


@dataclass(frozen=True)
class CorpusManifest:
    documents: list[DocumentRecord]


@dataclass(frozen=True)
class ProvenanceSegment:
    origin: str
    period: str
    licence: str
    licence_evidence: str
    credit: str


@dataclass(frozen=True)
class DatasetRecord:
    id: str
    licence: str
    url: str
    attribution_text: str
    permitted_context: str
    in_chart_pack: bool
    sha256: str
    human_signoff: HumanSignoff
    licence_evidence: str | None = None
    licence_note: str | None = None
    provenance: tuple[ProvenanceSegment, ...] = ()


@dataclass(frozen=True)
class Rebaseline:
    apply_to: str
    alignment_period_ce: tuple[Any, ...]
    display_reference: str | None = None
    rationale: str | None = None


@dataclass(frozen=True)
class SplicePair:
    id: str
    paleo: str
    instrumental: str
    splice_year_ce: Any
    rationale: str | None
    rebaseline: Rebaseline | None
    resolution_note: str | None = None


@dataclass(frozen=True)
class DatasetManifest:
    datasets: dict[str, DatasetRecord]
    splice_pairs: list[SplicePair]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _parse_date(value: Any, field_name: str, violations: list[str]) -> datetime.date | None:
    if isinstance(value, datetime.date):
        return value
    try:
        return datetime.date.fromisoformat(str(value))
    except ValueError:
        violations.append(f"{field_name} is not a valid ISO date: {value!r}")
        return None


def _missing(violations: list[str], field: str) -> None:
    violations.append(f"missing required field {field!r}")


def _validate_permitted_context(
    entry: Mapping[str, Any], violations: list[str], valid_contexts: frozenset[str]
) -> str | None:
    permitted_context = entry.get("permitted_context")
    choices = ", ".join(sorted(valid_contexts))
    if not permitted_context:
        violations.append(f"permitted_context is required (one of: {choices})")
    elif permitted_context not in valid_contexts:
        violations.append(
            f"permitted_context {permitted_context!r} is not a valid value (one of: {choices})"
        )
    return permitted_context


def _validate_human_signoff(entry: Mapping[str, Any], violations: list[str]) -> HumanSignoff | None:
    signoff_raw = entry.get("human_signoff")
    if not signoff_raw:
        violations.append("human_signoff is required ({who, date, note})")
        return None
    who = signoff_raw.get("who")
    date_raw = signoff_raw.get("date")
    note = signoff_raw.get("note")
    sub_missing = [
        name for name, val in (("who", who), ("date", date_raw), ("note", note)) if not val
    ]
    if sub_missing:
        violations.append("human_signoff is missing required subfields: " + ", ".join(sub_missing))
        return None
    date_val = _parse_date(date_raw, "human_signoff.date", violations)
    return HumanSignoff(who=who, date=date_val, note=note)


# ---------------------------------------------------------------------------
# Corpus documents
# ---------------------------------------------------------------------------


def load_corpus_manifest(path: Path) -> CorpusManifest:
    """Load + validate a corpus manifest; return an object with ``.documents``.

    ``.documents`` is a list of typed document records (attribute access).
    Any entry violating a §2.1 invariant refuses the whole load with
    :class:`ManifestError`.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    documents_raw = raw.get("documents") or []
    return CorpusManifest(documents=[validate_document(entry) for entry in documents_raw])


def validate_document(entry: Mapping[str, Any]) -> DocumentRecord:
    """Validate one corpus-manifest document entry; return its typed record.

    Pure over the mapping. Raises :class:`ManifestError` naming the entry
    id and every violated field. This is the build gate: indexing calls
    this (via load_corpus_manifest) before any document is ingested.
    """
    entry_id = entry.get("id") or "<unknown>"
    violations: list[str] = []

    licence = entry.get("licence")
    if not licence:
        _missing(violations, "licence")

    licence_evidence = entry.get("licence_evidence")
    if not licence_evidence:
        violations.append("licence_evidence is required to back any licence claim")

    attribution_text = entry.get("attribution_text")
    if not attribution_text:
        _missing(violations, "attribution_text")

    canonical_url = entry.get("canonical_url")
    if not canonical_url:
        _missing(violations, "canonical_url")

    if "redistributable" not in entry or entry.get("redistributable") is None:
        _missing(violations, "redistributable")
    redistributable = bool(entry.get("redistributable"))

    permitted_context = _validate_permitted_context(entry, violations, DOCUMENT_PERMITTED_CONTEXTS)

    permission_evidence = entry.get("permission_evidence")
    if permitted_context == "permission-on-file" and not permission_evidence:
        violations.append(
            "permission_evidence is required when permitted_context is 'permission-on-file'"
        )

    consensus_position = entry.get("consensus_position") or "assessed"

    sha256 = entry.get("sha256")
    if not sha256:
        _missing(violations, "sha256")

    retrieved_at_raw = entry.get("retrieved_at")
    retrieved_at = None
    if not retrieved_at_raw:
        _missing(violations, "retrieved_at")
    else:
        retrieved_at = _parse_date(retrieved_at_raw, "retrieved_at", violations)

    source_tier = entry.get("source_tier")
    if not source_tier:
        _missing(violations, "source_tier")

    human_signoff = _validate_human_signoff(entry, violations)

    if violations:
        raise ManifestError(f"{entry_id}: " + "; ".join(violations))

    return DocumentRecord(
        id=entry_id,
        licence=licence,
        licence_evidence=licence_evidence,
        attribution_text=attribution_text,
        canonical_url=canonical_url,
        redistributable=redistributable,
        permitted_context=permitted_context,
        permission_evidence=permission_evidence,
        consensus_position=consensus_position,
        sha256=sha256,
        retrieved_at=retrieved_at,
        source_tier=source_tier,
        human_signoff=human_signoff,
        path=entry.get("path") or None,
        source_url=entry.get("source_url") or None,
    )


# ---------------------------------------------------------------------------
# Dataset manifest
# ---------------------------------------------------------------------------


def load_dataset_manifest(path: Path) -> DatasetManifest:
    """Load + validate a dataset manifest; return an object with

    ``.datasets`` (mapping id -> typed record) and ``.splice_pairs``
    (list of typed pair records). Refuses invalid entries/pairs with
    :class:`ManifestError`.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    datasets_raw = raw.get("datasets") or {}
    datasets = {
        ds_id: validate_dataset({**entry, "id": ds_id}) for ds_id, entry in datasets_raw.items()
    }
    splice_pairs_raw = raw.get("splice_pairs") or []
    splice_pairs = [validate_splice_pair(pair) for pair in splice_pairs_raw]
    return DatasetManifest(datasets=datasets, splice_pairs=splice_pairs)


def validate_dataset(entry: Mapping[str, Any]) -> DatasetRecord:
    """Validate one dataset entry (id included in the mapping); return its

    typed record. Pure. Raises :class:`ManifestError` naming the dataset
    id and every violated field.
    """
    entry_id = entry.get("id") or "<unknown>"
    violations: list[str] = []

    licence = entry.get("licence")
    if not licence:
        _missing(violations, "licence")

    url = entry.get("url")
    if not url:
        _missing(violations, "url")

    attribution_text = entry.get("attribution_text")
    if not attribution_text:
        _missing(violations, "attribution_text")

    permitted_context = _validate_permitted_context(entry, violations, DATASET_PERMITTED_CONTEXTS)

    licence_note = entry.get("licence_note")
    licence_evidence = entry.get("licence_evidence")
    if permitted_context == "open-provisional":
        if not licence_note:
            violations.append(
                "licence_note is required when permitted_context is 'open-provisional' "
                "(the evidence trail for an unconfirmed verdict)"
            )
    else:
        if not licence_evidence:
            violations.append("licence_evidence is required to back any licence claim")

    if "in_chart_pack" not in entry or entry.get("in_chart_pack") is None:
        _missing(violations, "in_chart_pack")
    in_chart_pack = bool(entry.get("in_chart_pack"))
    if in_chart_pack and permitted_context != "open":
        violations.append(
            f"in_chart_pack requires permitted_context 'open' (got {permitted_context!r}) — "
            "the chart data pack ships only confirmed-open datasets"
        )

    sha256 = entry.get("sha256")
    if not sha256:
        _missing(violations, "sha256")

    human_signoff = _validate_human_signoff(entry, violations)

    provenance_raw = entry.get("provenance") or []
    provenance: list[ProvenanceSegment] = []
    for index, segment in enumerate(provenance_raw):
        origin = segment.get("origin")
        period = segment.get("period")
        seg_licence = segment.get("licence")
        seg_evidence = segment.get("licence_evidence")
        credit = segment.get("credit")
        label = origin or f"segment #{index}"
        seg_missing = [
            name
            for name, val in (
                ("origin", origin),
                ("period", period),
                ("licence", seg_licence),
                ("licence_evidence", seg_evidence),
                ("credit", credit),
            )
            if not val
        ]
        if seg_missing:
            violations.append(
                f"provenance segment {label!r} is missing required fields: "
                + ", ".join(seg_missing)
            )
        else:
            provenance.append(
                ProvenanceSegment(
                    origin=origin,
                    period=period,
                    licence=seg_licence,
                    licence_evidence=seg_evidence,
                    credit=credit,
                )
            )

    if provenance and attribution_text:
        for segment in provenance:
            if segment.credit not in attribution_text:
                violations.append(
                    f"attribution_text does not credit provenance segment {segment.credit!r}"
                )

    if violations:
        raise ManifestError(f"{entry_id}: " + "; ".join(violations))

    return DatasetRecord(
        id=entry_id,
        licence=licence,
        url=url,
        attribution_text=attribution_text,
        permitted_context=permitted_context,
        in_chart_pack=in_chart_pack,
        sha256=sha256,
        human_signoff=human_signoff,
        licence_evidence=licence_evidence,
        licence_note=licence_note,
        provenance=tuple(provenance),
    )


def validate_splice_pair(pair: Mapping[str, Any]) -> SplicePair:
    """Validate one splice-pair entry; return its typed record.

    ADR-020: the rebaseline decision is fixed in the manifest — a pair
    must carry either an explicit ``rebaseline: null`` or a rebaseline
    block with ``alignment_period_ce``; an absent ``rebaseline`` key
    refuses with :class:`ManifestError`.
    """
    pair_id = pair.get("id") or "<unknown>"
    violations: list[str] = []

    paleo = pair.get("paleo")
    if not paleo:
        _missing(violations, "paleo")

    instrumental = pair.get("instrumental")
    if not instrumental:
        _missing(violations, "instrumental")

    splice_year_ce = pair.get("splice_year_ce")
    if splice_year_ce is None:
        _missing(violations, "splice_year_ce")

    rationale = pair.get("rationale")

    rebaseline_record: Rebaseline | None = None
    if "rebaseline" not in pair:
        violations.append(
            "rebaseline decision is not recorded (ADR-020): a splice pair must carry "
            "either an explicit `rebaseline: null` or a block with alignment_period_ce"
        )
    else:
        rebaseline_raw = pair["rebaseline"]
        if rebaseline_raw is not None:
            apply_to = rebaseline_raw.get("apply_to")
            alignment_period_ce = rebaseline_raw.get("alignment_period_ce")
            if not apply_to:
                violations.append("rebaseline.apply_to is required")
            if not alignment_period_ce or len(alignment_period_ce) != 2:
                violations.append("rebaseline.alignment_period_ce must be a [start, end] pair")
            else:
                rebaseline_record = Rebaseline(
                    apply_to=apply_to,
                    alignment_period_ce=tuple(alignment_period_ce),
                    display_reference=rebaseline_raw.get("display_reference"),
                    rationale=rebaseline_raw.get("rationale"),
                )

    if violations:
        raise ManifestError(f"{pair_id}: " + "; ".join(violations))

    return SplicePair(
        id=pair_id,
        paleo=paleo,
        instrumental=instrumental,
        splice_year_ce=splice_year_ce,
        rationale=rationale,
        rebaseline=rebaseline_record,
        resolution_note=pair.get("resolution_note"),
    )


# ---------------------------------------------------------------------------
# Repo-shipping / ADR-023 checks
# ---------------------------------------------------------------------------


def check_prepared_text_shipping(documents: Iterable[Mapping[str, Any]], corpus_dir: Path) -> None:
    """The repo-shipping check for corpus text (DESIGN §2.1, runs in CI's

    unit stage): raise :class:`ManifestError` if prepared text exists under
    ``corpus_dir`` for any document whose ``permitted_context`` is not
    ``open``; return None when clean.
    """
    corpus_dir = Path(corpus_dir)
    violations = []
    for doc in documents:
        permitted_context = doc.get("permitted_context")
        path = doc.get("path")
        if permitted_context != "open" and path:
            candidate = corpus_dir / path
            if candidate.is_file():
                violations.append(
                    f"{doc.get('id', '<unknown>')}: prepared text committed at {path} for a "
                    f"non-open document (permitted_context={permitted_context!r})"
                )
    if violations:
        raise ManifestError("; ".join(violations))


def _has_synthetic_marker(path: Path) -> bool:
    try:
        with path.open(encoding="utf-8-sig", errors="strict") as handle:
            first_line = handle.readline()
    except (UnicodeDecodeError, OSError):
        return False
    return SYNTHETIC_FIXTURE_MARKER in first_line


def find_committed_data_files(repo_root: Path, tracked_files: Iterable[str]) -> list[str]:
    """The ADR-023 no-dataset-files-in-git check (runs in CI's unit stage).

    Given the repo root and git-tracked relative paths, return the tracked
    data-like files (at minimum: .csv/.tsv/.txt/.nc/.parquet) whose first
    line does not contain :data:`SYNTHETIC_FIXTURE_MARKER`. Files in
    :data:`_ADR023_OPERATIONAL_ALLOWLIST` (operational telemetry mandated for
    commit by other specs, e.g. the spend ledger) are exempt. An empty list
    means the tree is clean.
    """
    repo_root = Path(repo_root)
    offenders = []
    for rel_path in tracked_files:
        if rel_path in _ADR023_OPERATIONAL_ALLOWLIST:
            continue  # operational telemetry, not a dataset (see the constant's note)
        if Path(rel_path).suffix.lower() not in _DATA_LIKE_SUFFIXES:
            continue
        full_path = repo_root / rel_path
        if not full_path.is_file():
            continue
        if not _has_synthetic_marker(full_path):
            offenders.append(rel_path)
    return offenders


def verify_fetched_sha256(entry_id: str, path: Path, expected_sha256: str) -> None:
    """Verify a fetched file against its manifest-pinned hash (ADR-023).

    Return None on a match; raise :class:`ManifestError` naming
    ``entry_id`` and ``sha256`` on a mismatch or missing file.
    """
    path = Path(path)
    if not path.is_file():
        raise ManifestError(f"{entry_id}: sha256 verification failed — file not found at {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ManifestError(
            f"{entry_id}: sha256 mismatch (expected {expected_sha256}, got {digest})"
        )
