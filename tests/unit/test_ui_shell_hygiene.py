"""Issue #18 — structural guards on the UI's pure-core/imperative-shell split.

These are guards, not behavioural reds: they hold the issue's
architecture in place from the first commit (IMPLEMENTATION.md §1 — the
Streamlit file is the ONLY shell; every decision is pure and importable
without streamlit, httpx, FastAPI or the service app), and pin the UI's
wire-vocabulary constants EQUAL to the service's so the two cannot
drift (the UI deliberately does not import ``service.app``, which would
drag FastAPI into the UI image).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

UI_DIR = Path(__file__).resolve().parents[2] / "ui"

#: Every ui module except the Streamlit shell must be pure (no UI
#: framework, no network client, no service app / FastAPI).
PURE_MODULES = ("sse_client", "render_model", "starter", "footer", "charts", "presenters")

#: Modules that must never be imported (directly) by the pure core.
FORBIDDEN_IMPORTS = {
    "streamlit",
    "httpx",
    "requests",
    "aiohttp",
    "urllib.request",
    "fastapi",
    "service.app",
}


def imported_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


@pytest.mark.parametrize("module_name", PURE_MODULES)
def test_pure_ui_module_imports_no_shell_or_network_dependency(module_name: str) -> None:
    imports = imported_names(UI_DIR / f"{module_name}.py")
    offending = {
        name
        for name in imports
        if name in FORBIDDEN_IMPORTS
        or any(name.startswith(f"{banned}.") for banned in FORBIDDEN_IMPORTS)
    }
    assert not offending, (
        f"ui/{module_name}.py imports shell/network dependencies {sorted(offending)}; "
        "the pure core must be importable without them (IMPLEMENTATION.md §1)"
    )


def test_presenters_facade_exposes_the_whole_pure_contract() -> None:
    """IMPLEMENTATION.md §1 names ui/presenters.py as the decision surface;
    the shell composes from this one import."""
    import ui.presenters as presenters

    for name in (
        "parse_sse_stream",
        "stream_chat_events",
        "fold_chat_stream",
        "build_citation_chips",
        "chips_for_cached_citations",
        "source_list",
        "chat_page_model",
        "calibrated_term_anchors",
        "starter_groups",
        "starter_submission",
        "landing_page_model",
        "build_page_footer",
        "render_footer_lines",
        "chart_view_from_event",
    ):
        assert hasattr(presenters, name), f"ui.presenters does not export {name}"


class TestWireVocabularyParity:
    """The UI's constants equal the service's — drift fails here, not live."""

    def test_service_event_names_match(self) -> None:
        import service.app as service_app
        import ui.render_model as render_model

        assert render_model.META_EVENT == service_app.META_EVENT
        assert render_model.ANSWER_EVENT == service_app.ANSWER_EVENT
        assert render_model.CHART_EVENT == service_app.CHART_EVENT

    def test_answer_kind_values_match(self) -> None:
        import service.app as service_app
        import ui.render_model as render_model

        assert render_model.VIEW_KIND_CANNED == service_app.ANSWER_KIND_CANNED
        assert render_model.VIEW_KIND_REFUSAL == service_app.ANSWER_KIND_REFUSAL
        assert render_model.VIEW_KIND_PAUSED == service_app.ANSWER_KIND_PAUSED
        assert render_model.VIEW_KIND_CACHED_STARTER == service_app.ANSWER_KIND_CACHED_STARTER

    def test_validator_event_names_match(self) -> None:
        import ui.render_model as render_model
        from rag.citation_validator import (
            BADGE_EVENT,
            UNVERIFIED_REASON_ENTAILMENT,
            UNVERIFIED_REASON_UNCITED,
            VALIDATION_DEGRADED_EVENT,
        )

        assert render_model.BADGE_EVENT == BADGE_EVENT
        assert render_model.VALIDATION_DEGRADED_EVENT == VALIDATION_DEGRADED_EVENT
        # The badge reasons the view model displays are the validator's.
        assert UNVERIFIED_REASON_ENTAILMENT == "entailment_failed"
        assert UNVERIFIED_REASON_UNCITED == "uncited"
