"""Review-19 RED — issue #249: a live deploy must never silently serve
the interim placeholder transparency pages.

``service.main._build_transparency_pages`` returns ``None`` when
``evals/RESULTS.md`` is absent, keeping the pre-#19 placeholders in
``service.app`` — and ``validate_deployment_artifacts`` (#214/#216)
never checks RESULTS.md, so a fully LIVE deploy boots green and serves
pages with no attributions, no licence statements and an understated
privacy notice, forever. That inverts issue #19's ratified decision 4
("missing file fails the build") exactly where it matters.

Pinned fix, three parts:

- **Boot refusal:** ``validate_deployment_artifacts`` gains
  ``eval_results_path``; a live (index-recorded) deploy with a
  missing/unreadable RESULTS.md refuses at BOOT, naming the path
  alongside every other offender.
- **The dev tolerance survives:** ``index_corpus_version is None`` (the
  un-ingested dev/compose-smoke stack — the #215 zero-config boundary)
  still boots without RESULTS.md.
- **The placeholder state is honest:** the interim pages the dev stack
  legitimately serves carry the ADR-018 credit/non-commercial pair, the
  §4.11 disclaimer, an explicit interim marker, and no hand-coded
  retention figure that can silently diverge from the constants.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import service.app
import service.exchange_log
import service.main
import service.rate_limit
from service.app import ServiceStartupError
from service.transparency import (
    CREDIT_PAIR_MAX_SEPARATION,
    NON_AFFILIATION_DISCLAIMER,
    NONCOMMERCIAL_NOTE,
    STEWARD_CREDIT_TEXT,
)
from tests._transparency_fixtures import chars_between, contains_verbatim, page_text

LIVE_INDEX_VERSION = "corpus-2026-08-01"

#: The exact interim marker every placeholder page must carry — the
#: placeholder state has to be distinguishable ON THE ARTEFACT.
INTERIM_MARKER = "pre-release placeholder page"


def artifact_env(tmp_path: Path) -> dict[str, str]:
    """Readable stand-ins for the #214/#216 deployment artifacts."""
    manifest = tmp_path / "dataset-manifest.yaml"
    manifest.write_text("datasets: []\n", encoding="utf-8")
    pack_dir = tmp_path / "chart-pack"
    pack_dir.mkdir(exist_ok=True)
    threshold = tmp_path / "threshold.json"
    threshold.write_text("{}", encoding="utf-8")
    return {
        service.main.ENV_DATASET_MANIFEST: str(manifest),
        service.main.ENV_CHART_PACK_DIR: str(pack_dir),
        service.main.ENV_THRESHOLD_ARTIFACT: str(threshold),
    }


def readable_results(tmp_path: Path) -> Path:
    results = tmp_path / "RESULTS.md"
    results.write_text("Release verdict: PASSED\n", encoding="utf-8")
    return results


class TestLiveDeployRequiresEvalResults:
    """#249 boot rule: LIVE intent (per #216's live-prereq semantics —
    ``index_corpus_version`` is not None) requires readable published
    eval results; the placeholder-serving state is permitted ONLY for
    the dev/compose-smoke context."""

    def test_live_deploy_without_eval_results_refuses_boot(self, tmp_path) -> None:
        """All #216 artifacts present, RESULTS.md missing: the live
        deploy refuses at boot, naming the RESULTS.md path."""
        missing = tmp_path / "no-such-RESULTS.md"
        with pytest.raises(ServiceStartupError) as excinfo:
            service.main.validate_deployment_artifacts(
                artifact_env(tmp_path),
                index_corpus_version=LIVE_INDEX_VERSION,
                stored_chart_specs=False,
                eval_results_path=missing,
            )
        assert str(missing) in str(excinfo.value)

    def test_missing_eval_results_is_named_alongside_every_other_offender(self, tmp_path) -> None:
        """The name-every-offender discipline: an empty env plus a
        missing RESULTS.md lists all four offenders in ONE refusal."""
        missing = tmp_path / "no-such-RESULTS.md"
        with pytest.raises(ServiceStartupError) as excinfo:
            service.main.validate_deployment_artifacts(
                {},
                index_corpus_version=LIVE_INDEX_VERSION,
                stored_chart_specs=False,
                eval_results_path=missing,
            )
        message = str(excinfo.value)
        assert service.main.ENV_THRESHOLD_ARTIFACT in message
        assert service.main.ENV_DATASET_MANIFEST in message
        assert service.main.ENV_CHART_PACK_DIR in message
        assert str(missing) in message

    def test_unindexed_dev_stack_tolerates_missing_eval_results(self, tmp_path) -> None:
        """The #215 zero-config boundary survives: no recorded index
        (dev compose before ingestion) boots without RESULTS.md — the
        same tolerance as the un-ingested-index read-only start."""
        service.main.validate_deployment_artifacts(
            {},
            index_corpus_version=None,
            stored_chart_specs=False,
            eval_results_path=tmp_path / "no-such-RESULTS.md",
        )

    def test_live_deploy_with_readable_eval_results_passes(self, tmp_path) -> None:
        service.main.validate_deployment_artifacts(
            artifact_env(tmp_path),
            index_corpus_version=LIVE_INDEX_VERSION,
            stored_chart_specs=True,
            eval_results_path=readable_results(tmp_path),
        )

    def test_live_boot_without_eval_results_fails_at_create_service_app(
        self, monkeypatch, tmp_path
    ) -> None:
        """The wiring, end to end at this tier: a full deploy env with
        every #216 artifact readable and a live index recorded, but no
        evals/RESULTS.md — create_service_app must refuse naming the
        RESULTS.md path, instead of today's green boot that serves the
        interim placeholders indefinitely."""
        from tests._service_fixtures import (
            CORPUS_VERSION,
            apply_deploy_env,
            full_deploy_env,
        )

        env = full_deploy_env(tmp_path)
        env.update(artifact_env(tmp_path))
        apply_deploy_env(monkeypatch, env)
        monkeypatch.setattr(
            service.main,
            "_make_index_version_reader",
            lambda config: lambda: CORPUS_VERSION,
        )
        missing = tmp_path / "no-such-RESULTS.md"
        monkeypatch.setattr(service.main, "_EVAL_RESULTS_PATH", missing)

        with pytest.raises(ServiceStartupError) as excinfo:
            service.main.create_service_app()
        assert str(missing) in str(excinfo.value)


#: name -> the interim placeholder HTML constant in service.app (the
#: pages the dev/compose-smoke stack legitimately serves).
PLACEHOLDER_PAGES = {
    "about": service.app._ABOUT_HTML,
    "privacy": service.app._PRIVACY_HTML,
    "sources": service.app._SOURCES_HTML,
    "voices": service.app._VOICES_HTML,
}


class TestInterimPlaceholderHonesty:
    """#249: even the legitimate dev-state placeholders must not be
    materially dishonest — they carry the every-page invariants and say
    on their face that they are interim."""

    @pytest.mark.parametrize("name", sorted(PLACEHOLDER_PAGES))
    def test_interim_placeholder_carries_the_adr018_pair(self, name: str) -> None:
        """The ADR-018 credit/non-commercial pair, adjacent — an
        every-page invariant on the real pages, and the placeholders
        are still public pages."""
        text = page_text(PLACEHOLDER_PAGES[name])
        assert STEWARD_CREDIT_TEXT in text, f"placeholder /{name} lacks the steward credit"
        assert NONCOMMERCIAL_NOTE in text, f"placeholder /{name} lacks the non-commercial note"
        assert (
            chars_between(text, STEWARD_CREDIT_TEXT, NONCOMMERCIAL_NOTE)
            <= CREDIT_PAIR_MAX_SEPARATION
        ), f"placeholder /{name}: the ADR-018 pair is separated"

    @pytest.mark.parametrize("name", sorted(PLACEHOLDER_PAGES))
    def test_interim_placeholder_carries_the_nonaffiliation_disclaimer(self, name: str) -> None:
        assert contains_verbatim(PLACEHOLDER_PAGES[name], NON_AFFILIATION_DISCLAIMER), (
            f"placeholder /{name} is missing the §4.11 disclaimer verbatim"
        )

    @pytest.mark.parametrize("name", sorted(PLACEHOLDER_PAGES))
    def test_interim_placeholder_says_it_is_interim(self, name: str) -> None:
        """The placeholder state must be detectable from the outside:
        every placeholder page carries the explicit interim marker."""
        assert INTERIM_MARKER in page_text(PLACEHOLDER_PAGES[name]).lower(), (
            f"placeholder /{name} does not say it is a {INTERIM_MARKER!r}"
        )

    def test_placeholder_privacy_hard_codes_no_divergable_retention_figure(self) -> None:
        """The hand-written "no more than seven days" silently diverges
        the moment IP_HASH_RETENTION_DAYS changes (the real page
        interpolates at call time for exactly this reason): the
        placeholder either drops the figure or interpolates the
        constant — never a hand-copied number."""
        text = page_text(service.app._PRIVACY_HTML).lower()
        assert "seven days" not in text, (
            "placeholder /privacy hand-codes a spelled-out retention figure "
            "that can silently diverge from service.rate_limit.IP_HASH_RETENTION_DAYS"
        )
        allowed_figures = {
            service.rate_limit.IP_HASH_RETENTION_DAYS,
            service.exchange_log.EXCHANGE_LOG_RETENTION_DAYS,
        }
        for figure in re.findall(r"(\d+) days", text):
            assert int(figure) in allowed_figures, (
                f"placeholder /privacy carries a retention figure of {figure} days "
                "that matches no retention constant"
            )
