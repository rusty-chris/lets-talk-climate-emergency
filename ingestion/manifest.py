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
import json
import re
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

#: The CLOSED ``source_type`` vocabulary (review finding #158, ingestion
#: half). DESIGN §2.5/§3.2 define exactly two source layers: the RAG
#: evidence corpus (``evidence``) and the first-party voices layer
#: (``voices``). This frozenset is the SINGLE declaration for the whole
#: repo: it lives here (the dependency-light licensing gate every layer
#: already consumes) rather than in ``rag.indexing`` because importing it
#: the other way would couple ``ingestion`` -> ``rag`` and drag
#: ``qdrant_client`` through a cycle (``rag.indexing`` imports
#: ``ingestion.pipeline`` which imports this module). ``rag.indexing``
#: must re-export this very object — the red suite pins identity, never
#: a second copy. Matching is EXACT and case-sensitive (fail-closed, the
#: §2.1 include-list philosophy): a miscased, padded, or unknown value
#: refuses at manifest load, before any fetch, chunk, or index write.
SOURCE_TYPES = frozenset({"evidence", "voices"})

#: The §2.1 consensus flag is a closed enum (review #79): the §2.3
#: severity-skew guardrail keys on the exact string, so unknown values
#: refuse rather than silently reading as ``assessed``.
CONSENSUS_POSITIONS = frozenset({"assessed", "beyond-assessed-range"})

#: ``ingest_profile`` is a closed enum too (review #142 — the #79 defect
#: class on a new field): the whole Tier C headline-statements
#: protection (feature flag, ≤10-per-SPM cap, one-statement-per-chunk)
#: keys on the exact string, and the failure direction of a typo is
#: OPEN — the document would ingest as ordinary evidence with the flag
#: ignored. Absent means the ordinary evidence profile.
INGEST_PROFILES = frozenset({"headline-statements"})

#: First-line marker that exempts a committed data-like file from the
#: ADR-023 no-dataset-files-in-git check (same marker the #24 fixture
#: corpus uses; match on this substring, tolerating a BOM and either dash).
SYNTHETIC_FIXTURE_MARKER = "SYNTHETIC FIXTURE"

#: Second recognized first-line marker (review #83 / PR #93): a
#: *first-party operational record* — e.g. the committed spend ledger the
#: cost plan requires — is not licensed source data, ADR-023's target.
#: A distinct guarantee from :data:`SYNTHETIC_FIXTURE_MARKER`: that one
#: asserts invented test data; this one asserts a record this project
#: authored about its own operations, with no external licence attached.
#: Neither marker satisfies the other's check.
PROJECT_OPERATIONAL_DATA_MARKER = (
    "PROJECT-OPERATIONAL DATA — first-party record, no external licence"
)

#: Extensions treated as "data-like" for the ADR-023 no-committed-data
#: check. Extended per review #83 to the formats the MVP pack's providers
#: actually ship (OWID distributes .json/.xlsx variants; origin archives
#: serve gzipped CSVs); matched against the full multi-suffix chain so
#: ``real_data.csv.gz`` is caught.
_DATA_LIKE_SUFFIXES = frozenset(
    {
        ".csv",
        ".tsv",
        ".txt",
        ".nc",
        ".parquet",
        ".json",
        ".jsonl",
        ".xlsx",
        ".xls",
        ".dat",
        ".gz",
        ".zip",
        ".feather",
        ".arrow",
    }
)


class ManifestError(ValueError):
    """A licensing-invariant violation. The message names the offending

    document/dataset/pair id and every violated field (issue #5 acceptance
    criterion: refusal messages name the offending document and field).
    """


class Sha256MismatchError(ManifestError):
    """Fetched or committed bytes do not match the manifest-pinned sha256

    (ADR-023). A subclass so callers can distinguish upstream drift —
    which needs a re-pinning review — from a licensing refusal (review
    finding #81), while every existing ``except ManifestError`` still
    fails closed.
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
    #: Closed enum (review #142): None (ordinary evidence) or
    #: "headline-statements" — validated, never raw YAML.
    ingest_profile: str | None = None
    #: Closed enum (finding #158, ingestion half): a member of
    #: :data:`SOURCE_TYPES`, validated fail-closed at manifest load —
    #: missing, null, non-string, miscased, or unknown values refuse.
    #: Default None only satisfies the dataclass signature; every record
    #: :func:`validate_document` returns carries the validated string.
    source_type: str | None = None


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
    """A validated dataset entry.

    §2.1 field carriage for datasets (decided under review finding #78,
    per DESIGN §3 "every dataset carries the manifest fields from 2.1"):
    datasets carry ``licence``/``licence_evidence`` (or ``licence_note``
    for ``open-provisional``), ``attribution_text``, ``permitted_context``,
    ``permission_evidence`` (required when ``permission-on-file``),
    ``sha256``, ``retrieved_at`` (required — ADR-023's fetch-at-build
    model pins *which bytes*, and the retrieval date is part of that
    evidence) and ``human_signoff``. Deliberately NOT carried:
    ``canonical_url`` (its dataset analogue is ``url``), ``redistributable``
    (dataset redistribution is fixed by ADR-023 — never committed or
    mirrored — and ``in_chart_pack`` is the only shipping channel, gated
    on confirmed-open) and ``consensus_position`` (a claims-severity flag
    on documents; the §2.3 severity-skew guardrail reads document
    records, not numeric series).
    """

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
    permission_evidence: Any = None
    retrieved_at: datetime.date | None = None


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


def _require_date_field(
    entry: Mapping[str, Any], field: str, violations: list[str]
) -> datetime.date | None:
    """Require ``field`` to carry an ISO date, appending to ``violations``

    when it is absent or unparseable — the shared require-then-parse-date
    block behind ``retrieved_at`` on both documents and datasets.
    """
    raw = entry.get(field)
    if not raw:
        _missing(violations, field)
        return None
    return _parse_date(raw, field, violations)


#: The pin is bytes, not a promise: exactly 64 lowercase hex characters
#: (review #82 — 'TBD' pins nothing).
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _strip_or_none(value: Any) -> Any:
    """Strip a string value; whitespace-only becomes None (review #82)."""
    if isinstance(value, str):
        value = value.strip()
    return value or None


def _require_str(
    entry: Mapping[str, Any],
    field: str,
    violations: list[str],
    message: str | None = None,
) -> str | None:
    """Require a non-empty string field, stripping before the emptiness

    test (review #82: a single space satisfies no invariant). Returns the
    stripped value, or None after appending a violation.
    """
    value = entry.get(field)
    if isinstance(value, str):
        value = value.strip()
    if value is None or value == "":
        violations.append(message or f"missing required field {field!r}")
        return None
    if not isinstance(value, str):
        violations.append(f"{field} must be a non-empty string (got {type(value).__name__})")
        return None
    return value


def _require_bool(entry: Mapping[str, Any], field: str, violations: list[str]) -> bool:
    """Require a strictly typed boolean (review #82: the string 'false'

    coerces to True under bool() — the wrong-way flip for a legal flag).
    """
    value = entry.get(field)
    if value is None:
        _missing(violations, field)
        return False
    if not isinstance(value, bool):
        violations.append(f"{field} must be a boolean true/false (got {value!r})")
        return False
    return value


def _require_sha256(entry: Mapping[str, Any], violations: list[str]) -> str | None:
    sha256 = _require_str(entry, "sha256", violations)
    if sha256 is not None and not _SHA256_RE.fullmatch(sha256):
        violations.append(
            f"sha256 must be exactly 64 lowercase hex characters pinning the bytes "
            f"(ADR-023); got {sha256!r}"
        )
        return None
    return sha256


def _require_id(entry: Mapping[str, Any], violations: list[str]) -> str:
    entry_id = entry.get("id")
    if isinstance(entry_id, str):
        entry_id = entry_id.strip()
    if not entry_id or not isinstance(entry_id, str):
        # Review #82: an anonymous entry cannot be named by any refusal
        # and two anonymous records could coexist — id is required.
        _missing(violations, "id")
        return "<unknown>"
    return entry_id


def _validate_source_type(entry: Mapping[str, Any], violations: list[str]) -> str | None:
    """Closed enum (finding #158, ingestion half): every document names

    which of DESIGN §2.5/§3.2's two source layers it belongs to —
    ``evidence`` (the RAG corpus) or ``voices`` (the first-party
    layer). No default (missing or null refuses — a voices document
    that omits the field must never silently fall into the evidence
    corpus, past the #11 include-list) and no normalisation (a miscased
    or whitespace-padded value refuses rather than being coerced — the
    raw entry is what ``chunk_document`` later reads, so a normalised
    manifest would still leak the offending value onto chunk payloads).
    Matching is exact and case-sensitive, mirroring the #170 indexer/
    query-side enforcement of the very same vocabulary.
    """
    choices = ", ".join(sorted(SOURCE_TYPES))
    if "source_type" not in entry or entry.get("source_type") is None:
        violations.append(f"source_type is required (one of: {choices})")
        return None
    source_type = entry.get("source_type")
    if not isinstance(source_type, str):
        violations.append(f"source_type must be a string (one of: {choices}); got {source_type!r}")
        return None
    if source_type not in SOURCE_TYPES:
        violations.append(f"source_type {source_type!r} is not a valid value (one of: {choices})")
        return None
    return source_type


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


def _validate_permission_evidence(
    entry: Mapping[str, Any], permitted_context: Any, violations: list[str]
) -> Any:
    """The Tier-C rule, shared by documents and datasets (review #78):

    ``permission-on-file`` is a claim about a letter on file, so it
    requires non-empty ``permission_evidence``.
    """
    permission_evidence = entry.get("permission_evidence")
    if isinstance(permission_evidence, str):
        permission_evidence = permission_evidence.strip() or None
    if permitted_context == "permission-on-file" and not permission_evidence:
        violations.append(
            "permission_evidence is required when permitted_context is 'permission-on-file'"
        )
    return permission_evidence


def _validate_human_signoff(entry: Mapping[str, Any], violations: list[str]) -> HumanSignoff | None:
    signoff_raw = entry.get("human_signoff")
    if not signoff_raw:
        violations.append("human_signoff is required ({who, date, note})")
        return None
    if not isinstance(signoff_raw, Mapping):
        # Review #82: a wrong-typed sub-structure is a naming refusal,
        # never an AttributeError traceback.
        violations.append(
            f"human_signoff must be a mapping with who/date/note (got {type(signoff_raw).__name__})"
        )
        return None
    who = signoff_raw.get("who")
    date_raw = signoff_raw.get("date")
    note = signoff_raw.get("note")
    if isinstance(who, str):
        who = who.strip()
    if isinstance(note, str):
        note = note.strip()
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
    documents_raw = raw.get("documents") if isinstance(raw, Mapping) else None
    if not isinstance(documents_raw, list):
        # Fail closed (review #77): a missing/typo'd/null `documents` key
        # must never load as an empty corpus and let the gate pass
        # vacuously. An empty corpus must say so: `documents: []`.
        raise ManifestError(
            f"corpus manifest at {path} carries no 'documents' list — refusing to treat a "
            "mis-keyed manifest as an empty corpus (an intentionally empty corpus must "
            "declare an explicit `documents: []`)"
        )
    return CorpusManifest(documents=[validate_document(entry) for entry in documents_raw])


def validate_document(entry: Mapping[str, Any]) -> DocumentRecord:
    """Validate one corpus-manifest document entry; return its typed record.

    Pure over the mapping. Raises :class:`ManifestError` naming the entry
    id and every violated field. This is the build gate: indexing calls
    this (via load_corpus_manifest) before any document is ingested.
    """
    violations: list[str] = []
    entry_id = _require_id(entry, violations)

    licence = _require_str(entry, "licence", violations)

    licence_evidence = _require_str(
        entry,
        "licence_evidence",
        violations,
        message="licence_evidence is required to back any licence claim",
    )

    attribution_text = _require_str(entry, "attribution_text", violations)

    canonical_url = _require_str(entry, "canonical_url", violations)

    redistributable = _require_bool(entry, "redistributable", violations)

    permitted_context = _validate_permitted_context(entry, violations, DOCUMENT_PERMITTED_CONTEXTS)

    permission_evidence = _validate_permission_evidence(entry, permitted_context, violations)

    source_type = _validate_source_type(entry, violations)

    # DESIGN §2.1 / TDD-plan item 10: an *absent* consensus_position (or an
    # explicit null) defaults to "assessed"; any present value — including
    # an explicit empty string — must be a member of the closed enum
    # (review #79: the fail-open direction here silently disarms the §2.3
    # severity-skew guardrail).
    consensus_position = entry.get("consensus_position")
    if consensus_position is None:
        consensus_position = "assessed"
    elif consensus_position not in CONSENSUS_POSITIONS:
        choices = ", ".join(sorted(CONSENSUS_POSITIONS))
        violations.append(
            f"consensus_position {consensus_position!r} is not a valid value (one of: {choices})"
        )

    sha256 = _require_sha256(entry, violations)

    retrieved_at = _require_date_field(entry, "retrieved_at", violations)

    source_tier = _require_str(entry, "source_tier", violations)

    human_signoff = _validate_human_signoff(entry, violations)

    # Review #142: `ingest_profile` is a closed enum (absent |
    # headline-statements). The Tier C protection keys on the exact
    # string and a typo fails OPEN (the curated set ingests as ordinary
    # evidence with the feature flag ignored), so any unknown value
    # refuses here — before any fetch, parse or chunk.
    ingest_profile = entry.get("ingest_profile")
    if ingest_profile is not None and ingest_profile not in INGEST_PROFILES:
        choices = ", ".join(sorted(INGEST_PROFILES))
        violations.append(
            f"ingest_profile {ingest_profile!r} is not a valid value "
            f"(absent for ordinary evidence, or one of: {choices})"
        )

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
        ingest_profile=ingest_profile,
        source_type=source_type,
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

    # Referential integrity across the manifest (review #84): a pair's
    # datasets — and the rebaseline target — must exist, or the ADR-020
    # decision is stranded and fails later as a KeyError far from the
    # cause (or worse, is improvised around).
    referential_violations = []
    for pair in splice_pairs:
        dangling = [
            reference for reference in (pair.paleo, pair.instrumental) if reference not in datasets
        ]
        for reference in dangling:
            referential_violations.append(
                f"{pair.id}: references unknown dataset id {reference!r} "
                "(not declared in this manifest)"
            )
    if referential_violations:
        raise ManifestError("; ".join(referential_violations))

    return DatasetManifest(datasets=datasets, splice_pairs=splice_pairs)


def validate_dataset(entry: Mapping[str, Any]) -> DatasetRecord:
    """Validate one dataset entry (id included in the mapping); return its

    typed record. Pure. Raises :class:`ManifestError` naming the dataset
    id and every violated field.
    """
    violations: list[str] = []
    entry_id = _require_id(entry, violations)

    licence = _require_str(entry, "licence", violations)

    url = _require_str(entry, "url", violations)

    attribution_text = _require_str(entry, "attribution_text", violations)

    permitted_context = _validate_permitted_context(entry, violations, DATASET_PERMITTED_CONTEXTS)

    permission_evidence = _validate_permission_evidence(entry, permitted_context, violations)

    licence_note = _strip_or_none(entry.get("licence_note"))
    licence_evidence = _strip_or_none(entry.get("licence_evidence"))
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
    in_chart_pack = _require_bool(entry, "in_chart_pack", violations)
    if in_chart_pack and permitted_context != "open":
        violations.append(
            f"in_chart_pack requires permitted_context 'open' (got {permitted_context!r}) — "
            "the chart data pack ships only confirmed-open datasets"
        )

    sha256 = _require_sha256(entry, violations)

    retrieved_at_raw = entry.get("retrieved_at")
    retrieved_at = None
    if not retrieved_at_raw:
        _missing(violations, "retrieved_at")
    else:
        retrieved_at = _parse_date(retrieved_at_raw, "retrieved_at", violations)

    human_signoff = _validate_human_signoff(entry, violations)

    provenance_raw = entry.get("provenance") or []
    provenance: list[ProvenanceSegment] = []
    if not isinstance(provenance_raw, (list, tuple)):
        # Review #82: wrong-typed sub-structures refuse by name, never
        # crash with AttributeError.
        violations.append(
            f"provenance must be a list of segment mappings (got {type(provenance_raw).__name__})"
        )
        provenance_raw = []
    for index, segment in enumerate(provenance_raw):
        if not isinstance(segment, Mapping):
            violations.append(
                f"provenance segment #{index} must be a mapping (got {type(segment).__name__})"
            )
            continue

        origin = _strip_or_none(segment.get("origin"))
        period = _strip_or_none(segment.get("period"))
        seg_licence = _strip_or_none(segment.get("licence"))
        seg_evidence = _strip_or_none(segment.get("licence_evidence"))
        credit = _strip_or_none(segment.get("credit"))
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
        permission_evidence=permission_evidence,
        retrieved_at=retrieved_at,
    )


def validate_splice_pair(pair: Mapping[str, Any]) -> SplicePair:
    """Validate one splice-pair entry; return its typed record.

    ADR-020: the rebaseline decision is fixed in the manifest — a pair
    must carry either an explicit ``rebaseline: null`` or a rebaseline
    block with ``alignment_period_ce``; an absent ``rebaseline`` key
    refuses with :class:`ManifestError`.
    """
    violations: list[str] = []
    pair_id = _require_id(pair, violations)

    paleo = _require_str(pair, "paleo", violations)

    instrumental = _require_str(pair, "instrumental", violations)

    splice_year_ce = pair.get("splice_year_ce")
    if splice_year_ce is None:
        _missing(violations, "splice_year_ce")

    rationale = _strip_or_none(pair.get("rationale"))

    rebaseline_record: Rebaseline | None = None
    if "rebaseline" not in pair:
        violations.append(
            "rebaseline decision is not recorded (ADR-020): a splice pair must carry "
            "either an explicit `rebaseline: null` or a block with alignment_period_ce"
        )
    else:
        rebaseline_raw = pair["rebaseline"]
        if rebaseline_raw is None and not rationale:
            # Review #84: the explicit-null (absolute-scale) form is a
            # decision and must say why — the documented contract has
            # always been "explicit `rebaseline: null` with rationale".
            violations.append(
                "an explicit `rebaseline: null` decision requires a rationale "
                "recording why no rebaseline is needed or legal for this pair"
            )
        if rebaseline_raw is not None and not isinstance(rebaseline_raw, Mapping):
            # Review #82: `rebaseline: "null"` (a string) is a YAML slip,
            # not a decision — refuse by name, never AttributeError.
            violations.append(
                "rebaseline must be a mapping with alignment_period_ce or an explicit "
                f"`rebaseline: null` (got {rebaseline_raw!r})"
            )
        elif rebaseline_raw is not None:
            apply_to = _strip_or_none(rebaseline_raw.get("apply_to"))
            alignment_period_ce = rebaseline_raw.get("alignment_period_ce")
            if not apply_to:
                violations.append("rebaseline.apply_to is required")
            elif apply_to not in {paleo, instrumental}:
                # Review #84: rebaselining a series that is not in the
                # pair is not a legal reading of the fixed decision.
                violations.append(
                    f"rebaseline.apply_to {apply_to!r} must name one of the pair's members "
                    f"({paleo!r} or {instrumental!r})"
                )
            if not isinstance(alignment_period_ce, (list, tuple)) or len(alignment_period_ce) != 2:
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


#: Housekeeping files legitimately living at the top of a corpus directory
#: — the manifest itself, its README, the hand-audit checklist, and the
#: written ingest run record (``ingest_run.json``, review #143: a
#: first-party operational record of the run, never licensed source
#: text). Everything else present must be the declared ``path`` of a
#: ``permitted_context: open`` document.
_SHIP_EXEMPT_NAMES = frozenset(
    {"manifest.yaml", "readme.md", "ingest_run.json", "hand-audit-checklist.md"}
)

#: Interpreter cache artefacts, never prepared text.
_CACHE_SUFFIXES = frozenset({".pyc", ".pyo"})


def check_prepared_text_shipping(documents: Iterable[Mapping[str, Any]], corpus_dir: Path) -> None:
    """The repo-shipping check for corpus text (DESIGN §2.1, runs in CI's

    unit stage): raise :class:`ManifestError` unless the corpus tree is
    fully accounted for; return None when clean. Two arms, both fail
    closed (review finding #77):

    1. No declared ``path`` of a non-``open`` document may resolve to a
       file under ``corpus_dir``.
    2. Every file under ``corpus_dir`` (housekeeping ``manifest.yaml``/
       ``README.md``, dotfiles and interpreter cache artefacts excepted)
       must be the declared ``path`` of a ``permitted_context: open``
       document — an undeclared file is exactly how non-open text would
       ship (committed under a scratch name, or with the manifest entry
       left ``path: null``), so it refuses by name.
    """
    corpus_dir = Path(corpus_dir)
    violations = []
    open_paths: set[str] = set()
    for doc in documents:
        permitted_context = doc.get("permitted_context")
        path = doc.get("path")
        if path and permitted_context == "open":
            open_paths.add(Path(str(path)).as_posix())
        if permitted_context != "open" and path:
            candidate = corpus_dir / path
            if candidate.is_file():
                violations.append(
                    f"{doc.get('id', '<unknown>')}: prepared text committed at {path} for a "
                    f"non-open document (permitted_context={permitted_context!r})"
                )
    if corpus_dir.is_dir():
        for candidate in sorted(corpus_dir.rglob("*")):
            if not candidate.is_file():
                continue
            rel = candidate.relative_to(corpus_dir)
            if "__pycache__" in rel.parts or candidate.suffix.lower() in _CACHE_SUFFIXES:
                continue
            if candidate.name.startswith("."):
                continue
            if len(rel.parts) == 1 and candidate.name.lower() in _SHIP_EXEMPT_NAMES:
                continue
            if rel.as_posix() not in open_paths:
                violations.append(
                    f"undeclared prepared text at {rel.as_posix()}: not the declared path of "
                    "any permitted_context 'open' document — non-open or unreviewed text "
                    "must never sit in the corpus tree (DESIGN §2.1)"
                )
    if violations:
        raise ManifestError("; ".join(violations))


def _has_exemption_marker(path: Path) -> bool:
    """True when the file's first line carries a recognized exemption

    marker. Undecodable (binary/compressed) content returns False — there
    is no marker mechanism inside such files, so they fail closed.
    """
    try:
        with path.open(encoding="utf-8-sig", errors="strict") as handle:
            first_line = handle.readline()
    except (UnicodeDecodeError, OSError):
        return False
    return SYNTHETIC_FIXTURE_MARKER in first_line or PROJECT_OPERATIONAL_DATA_MARKER in first_line


def _has_envelope_provenance(path: Path) -> bool:
    """True when a .json file carries the recorded-envelope provenance

    declaration (finding #67's convention, used by the replay fixtures):
    a top-level ``_meta`` mapping with a non-empty ``provenance`` string
    and a non-empty ``content_signoff`` mapping. Anything unparseable
    fails closed.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return False
    if not isinstance(payload, Mapping):
        return False
    meta = payload.get("_meta")
    if not isinstance(meta, Mapping):
        return False
    provenance = meta.get("provenance")
    signoff = meta.get("content_signoff")
    return bool(
        isinstance(provenance, str)
        and provenance.strip()
        and isinstance(signoff, Mapping)
        and signoff
    )


def find_committed_data_files(repo_root: Path, tracked_files: Iterable[str]) -> list[str]:
    """The ADR-023 no-dataset-files-in-git check (runs in CI's unit stage).

    Given the repo root and git-tracked relative paths, return the tracked
    data-like files (matched on the full multi-suffix chain against
    :data:`_DATA_LIKE_SUFFIXES`, so ``real_data.csv.gz`` is caught) that
    carry no recognized exemption. An empty list means the tree is clean.

    Two first-line markers exempt a text file, with distinct guarantees:

    - :data:`SYNTHETIC_FIXTURE_MARKER` — invented test data authored for
      this repo (the #24 fixture convention); asserts no real Tier B/C
      content.
    - :data:`PROJECT_OPERATIONAL_DATA_MARKER` — a first-party record
      about this project's own operations (e.g. the committed spend
      ledger, PR #93); asserts the data is ours, with no external
      licence. Not valid in fixture directories, whose meta-tests
      require the synthetic marker specifically.

    A ``.json`` file may instead carry the recorded-envelope provenance
    declaration (``_meta.provenance`` + ``_meta.content_signoff``, the
    finding-#67 convention of the replay fixtures). Compressed or binary
    data-like files have no exemption mechanism and are always offenders
    (review #83: fail closed).
    """
    repo_root = Path(repo_root)
    offenders = []
    for rel_path in tracked_files:
        suffix_chain = {suffix.lower() for suffix in Path(rel_path).suffixes}
        if not (suffix_chain & _DATA_LIKE_SUFFIXES):
            continue
        full_path = repo_root / rel_path
        if not full_path.is_file():
            continue
        if _has_exemption_marker(full_path):
            continue
        if full_path.suffix.lower() == ".json" and _has_envelope_provenance(full_path):
            continue
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
        raise Sha256MismatchError(
            f"{entry_id}: sha256 mismatch (expected {expected_sha256}, got {digest})"
        )
