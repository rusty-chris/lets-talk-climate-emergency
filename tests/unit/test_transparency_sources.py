"""Issue #19 RED — the /sources page is GENERATED from the manifests.

``corpus/manifest.yaml`` + ``datasets/manifest.yaml`` are the single
sources of truth (DESIGN §2.1/§3.7); the page is a projection of them:

- every ACTIVE document appears with title, manifest-VERBATIM
  attribution_text, licence and canonical_url, grouped by tier;
- ``permitted_context: non-commercial-educational`` entries carry their
  non-commercial note beside them;
- the commented pending-source skeleton NEVER appears (nothing is
  listed that is not real, pinned and signed off);
- every dataset appears with attribution, licence and fetch provenance
  (origin URL + access date; ADR-023 — fetched at build, hash-verified,
  never committed);
- a manifest ADDITION appears with zero code change (the tests render a
  manifest with a synthetic extra entry and find it on the page).

Review-19 fix pins (red-first):

- #250: ``in_chart_pack: false`` / ``permitted_context:
  open-provisional`` datasets are separated from the pack section and
  carry a pending marker; the page makes no completeness claim its own
  entries contradict; no "see provenance" reference dangles without the
  provenance actually rendered.
- #253: a document whose ``source_tier`` is outside the rendered order
  fails the build LOUDLY (``TransparencyBuildError`` naming the
  document), never silently vanishing from the attribution surface.

These pins are driven FROM the loaded manifests, never from hard-coded
source lists — a new source cannot be forgotten
(``test_every_required_attribution_string_appears``).
"""

from __future__ import annotations

import pytest

from service.transparency import TransparencyBuildError, render_sources_page
from tests._transparency_fixtures import (
    SYNTHETIC_NC_DOCUMENT,
    SYNTHETIC_NEW_DATASET,
    SYNTHETIC_NEW_DOCUMENT,
    chars_between,
    contains_verbatim,
    corpus_manifest_with,
    datasets_manifest_with,
    load_real_corpus_manifest,
    load_real_datasets_manifest,
    page_text,
)

CORPUS_VINTAGE = "2026-08-01"


def render_real() -> str:
    return render_sources_page(
        corpus_manifest=load_real_corpus_manifest(),
        datasets_manifest=load_real_datasets_manifest(),
        corpus_vintage=CORPUS_VINTAGE,
    )


class TestManifestDrivenAttribution:
    def test_every_required_attribution_string_appears(self) -> None:
        """For EVERY active document and EVERY dataset, the manifest's
        attribution_text appears on the page verbatim — driven from the
        manifests themselves, so a newly-added source (the C3S exact
        string, the OGL statement, dataset citations…) is covered the
        moment its entry lands."""
        rendered = render_real()
        for document in load_real_corpus_manifest()["documents"]:
            assert contains_verbatim(rendered, document["attribution_text"]), (
                f"document {document['id']}: attribution_text not manifest-verbatim on /sources"
            )
        for dataset_id, entry in load_real_datasets_manifest()["datasets"].items():
            assert contains_verbatim(rendered, entry["attribution_text"]), (
                f"dataset {dataset_id}: attribution_text not manifest-verbatim on /sources"
            )

    def test_every_document_lists_title_licence_and_canonical_url(self) -> None:
        rendered = render_real()
        for document in load_real_corpus_manifest()["documents"]:
            for field in ("title", "licence", "canonical_url"):
                assert contains_verbatim(rendered, document[field]), (
                    f"document {document['id']}: {field} missing from /sources"
                )

    def test_documents_are_grouped_by_tier(self) -> None:
        """Tier headings structure the page; each document renders under
        its manifest tier (position pin: heading before title)."""
        manifest = corpus_manifest_with(SYNTHETIC_NC_DOCUMENT)
        text = page_text(
            render_sources_page(
                corpus_manifest=manifest,
                datasets_manifest=load_real_datasets_manifest(),
                corpus_vintage=CORPUS_VINTAGE,
            )
        )
        assert "Tier A" in text and "Tier B" in text
        tier_a_at = text.index("Tier A")
        tier_b_at = text.index("Tier B")
        assert tier_a_at < tier_b_at
        for document in manifest["documents"]:
            title_at = text.index(" ".join(document["title"].split()))
            if document["source_tier"] == "A":
                assert tier_a_at < title_at < tier_b_at, (
                    f"{document['id']} (Tier A) not rendered under the Tier A heading"
                )
            else:
                assert title_at > tier_b_at, (
                    f"{document['id']} (Tier B) not rendered under the Tier B heading"
                )

    def test_noncommercial_entries_carry_their_nc_note(self) -> None:
        """permitted_context: non-commercial-educational is load-bearing
        (ADR-018) — the entry says so beside itself."""
        text = page_text(
            render_sources_page(
                corpus_manifest=corpus_manifest_with(SYNTHETIC_NC_DOCUMENT),
                datasets_manifest=load_real_datasets_manifest(),
                corpus_vintage=CORPUS_VINTAGE,
            )
        )
        title = " ".join(SYNTHETIC_NC_DOCUMENT["title"].split())
        assert title in text
        assert chars_between(text, title, "non-commercial") <= 400, (
            "the NC entry's non-commercial note is not beside the entry"
        )


class TestPendingSkeletonNeverListed:
    def test_commented_pending_sources_do_not_appear(self) -> None:
        """The manifest's pending skeleton is comments, not entries —
        nothing pending, unpinned or unsigned may masquerade as a source
        on the public page."""
        text = page_text(render_real())
        for pending_token in (
            "unep_egr",
            "hansen_2023_pipeline",
            "hansen_2025_acceleration",
            "carbon_brief_verbatim_set",
            "berkeley_earth_text",
            "ripple_bioscience_warnings",
            "UNEP Emissions Gap",
            "Carbon Brief",
            "Berkeley Earth",
        ):
            assert pending_token not in text, (
                f"/sources lists pending (not-yet-real) source {pending_token!r}"
            )


class TestAdditionsAppearWithoutCodeChange:
    def test_new_document_entry_appears(self) -> None:
        rendered = render_sources_page(
            corpus_manifest=corpus_manifest_with(SYNTHETIC_NEW_DOCUMENT),
            datasets_manifest=load_real_datasets_manifest(),
            corpus_vintage=CORPUS_VINTAGE,
        )
        assert contains_verbatim(rendered, SYNTHETIC_NEW_DOCUMENT["title"])
        assert contains_verbatim(rendered, SYNTHETIC_NEW_DOCUMENT["attribution_text"])
        assert contains_verbatim(rendered, SYNTHETIC_NEW_DOCUMENT["canonical_url"])

    def test_new_dataset_entry_appears(self) -> None:
        rendered = render_sources_page(
            corpus_manifest=load_real_corpus_manifest(),
            datasets_manifest=datasets_manifest_with(syn_basin_anomaly=SYNTHETIC_NEW_DATASET),
            corpus_vintage=CORPUS_VINTAGE,
        )
        assert contains_verbatim(rendered, SYNTHETIC_NEW_DATASET["attribution_text"])
        assert contains_verbatim(rendered, SYNTHETIC_NEW_DATASET["url"])


class TestDatasetProvenance:
    def test_every_dataset_lists_fetch_provenance(self) -> None:
        """ADR-023 made visible: each dataset names its origin URL, and
        the page carries the pack access date and the hash-verification
        story (fetched at build, verified against pinned sha256)."""
        rendered = render_real()
        manifest = load_real_datasets_manifest()
        for dataset_id, entry in manifest["datasets"].items():
            assert contains_verbatim(rendered, entry["url"]), (
                f"dataset {dataset_id}: origin URL missing from /sources"
            )
        text = page_text(rendered)
        assert manifest["access_date"] in text
        assert "sha256" in text.lower()
        assert "fetched" in text.lower()

    def test_no_dangling_provenance_references(self) -> None:
        """#250: licence texts rendered verbatim end "— see provenance
        below" while the manifest's ``provenance`` blocks are never
        rendered — a dangling reference on the public page. Every entry
        carrying a provenance block must render it (a "Provenance:"
        block beside the entry, with each segment's origin, period and
        credit); every see-provenance licence must have a block to
        point at. Manifest-driven — zero hard-coded dataset lists."""
        text = page_text(render_real())
        provenance_seen = False
        for dataset_id, entry in load_real_datasets_manifest()["datasets"].items():
            licence = " ".join(str(entry.get("licence", "")).split()).lower()
            if "see provenance" in licence:
                assert "provenance" in entry, (
                    f"dataset {dataset_id}: licence points at provenance the "
                    "manifest does not carry"
                )
            segments = entry.get("provenance") or []
            if not segments:
                continue
            provenance_seen = True
            attribution = " ".join(str(entry["attribution_text"]).split())
            assert chars_between(text, attribution, "Provenance:") <= 600, (
                f"dataset {dataset_id}: no rendered Provenance block beside the entry"
            )
            for segment in segments:
                for field in ("origin", "period", "credit"):
                    value = " ".join(str(segment[field]).split())
                    assert value in text, (
                        f"dataset {dataset_id}: provenance {field} {value!r} "
                        "not rendered on /sources"
                    )
        assert provenance_seen, (
            "the real manifest lost all provenance blocks; retire this pin deliberately"
        )


#: #250 contract strings (the implementer renders EXACTLY these).
PENDING_MARKER = "licence confirmation pending — not used in charts"
PACK_HEADING = "Chart datasets"
PROVISIONAL_HEADING = "Datasets under licence confirmation"
HONEST_COMPLETENESS_CLAIM = "entries awaiting written licence confirmation are marked as such"
RETIRED_COMPLETENESS_CLAIM = "nothing pending or unsigned is ever listed"


class TestProvisionalDatasetHonesty:
    """Review finding #250: bereiter2015_co2 and kaufman2020_temp12k are
    ``permitted_context: open-provisional`` / ``in_chart_pack: false`` —
    their own manifest records licence confirmation as PENDING (issue
    #23) and the pack invariant excludes them — yet the page rendered
    them indistinguishably under "Chart datasets", directly beneath
    "nothing pending or unsigned is ever listed". Every sentence on the
    page must be true of what is listed."""

    def test_provisional_datasets_carry_a_pending_marker(self) -> None:
        """Every ``permitted_context != "open"`` dataset carries the
        pending marker beside its attribution; pack datasets never do
        (the marker count equals the provisional count exactly)."""
        text = page_text(render_real())
        datasets = load_real_datasets_manifest()["datasets"]
        provisional = {
            dataset_id: entry
            for dataset_id, entry in datasets.items()
            if entry.get("permitted_context") != "open"
        }
        assert provisional, (
            "the real manifest lost its provisional entries; retire this pin deliberately"
        )
        assert text.count(PENDING_MARKER) == len(provisional), (
            f"expected the pending marker exactly once per provisional dataset "
            f"({len(provisional)}), found {text.count(PENDING_MARKER)}"
        )
        for dataset_id, entry in provisional.items():
            attribution = " ".join(str(entry["attribution_text"]).split())
            assert chars_between(text, attribution, PENDING_MARKER) <= 400, (
                f"dataset {dataset_id}: the pending marker is not beside its entry"
            )

    def test_chart_dataset_section_separates_pack_from_provisional(self) -> None:
        """Position pin (the test_documents_are_grouped_by_tier pattern):
        the section whose heading claims chart use contains ONLY
        ``in_chart_pack: true`` entries; ``in_chart_pack: false`` entries
        render under their own honest heading below it."""
        text = page_text(render_real())
        assert PROVISIONAL_HEADING in text, (
            f"/sources has no {PROVISIONAL_HEADING!r} section — provisional "
            "datasets still masquerade as chart-pack datasets"
        )
        pack_heading_at = text.index(PACK_HEADING)
        provisional_heading_at = text.index(PROVISIONAL_HEADING)
        assert pack_heading_at < provisional_heading_at
        for dataset_id, entry in load_real_datasets_manifest()["datasets"].items():
            attribution = " ".join(str(entry["attribution_text"]).split())
            attribution_at = text.index(attribution)
            if entry.get("in_chart_pack"):
                assert pack_heading_at < attribution_at < provisional_heading_at, (
                    f"dataset {dataset_id} (in the pack) is not under the {PACK_HEADING!r} heading"
                )
            else:
                assert attribution_at > provisional_heading_at, (
                    f"dataset {dataset_id} (NOT in the pack) renders under the "
                    "chart-use section — the page claims charts are built from it"
                )

    def test_sources_page_makes_no_false_completeness_claim(self) -> None:
        """The single most load-bearing honesty sentence: while
        open-provisional entries render, "nothing pending or unsigned is
        ever listed" is false on its own page. The reworded claim tells
        the truth about the marking."""
        text = page_text(render_real())
        assert RETIRED_COMPLETENESS_CLAIM not in text, (
            "/sources still claims nothing pending is ever listed while "
            "rendering entries whose own licence text says NOT confirmed"
        )
        assert HONEST_COMPLETENESS_CLAIM in text


class TestUnknownTierFailsLoudly:
    """Review finding #253: a document with a tier outside
    ``_SOURCE_TIER_ORDER`` was grouped under its unknown key and then
    silently never rendered — an active, cited document vanishing from
    the public attribution surface with no failure anywhere."""

    def test_document_with_unlisted_tier_is_never_silently_dropped(self) -> None:
        """Pinned resolution: the build fails LOUDLY —
        TransparencyBuildError naming the document id and the unexpected
        tier value — so a tier typo can never cost an attribution."""
        odd_tier_document = {
            **SYNTHETIC_NEW_DOCUMENT,
            "id": "syn_odd_tier",
            "source_tier": "D",
        }
        with pytest.raises(TransparencyBuildError) as excinfo:
            render_sources_page(
                corpus_manifest=corpus_manifest_with(odd_tier_document),
                datasets_manifest=load_real_datasets_manifest(),
                corpus_vintage=CORPUS_VINTAGE,
            )
        message = str(excinfo.value)
        assert "syn_odd_tier" in message, "the offending document is not named"
        assert "'D'" in message, "the unexpected tier value is not named"
