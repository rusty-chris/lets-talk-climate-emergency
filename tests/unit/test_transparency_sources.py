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

These pins are driven FROM the loaded manifests, never from hard-coded
source lists — a new source cannot be forgotten
(``test_every_required_attribution_string_appears``).
"""

from __future__ import annotations

from service.transparency import render_sources_page
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
