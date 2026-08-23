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
        # Review finding #224: the transport-failure honesty path is pure
        # and shell-reachable through the one facade.
        "transport_failure_view",
        "answer_status_lines",
        "TransportError",
        # Review finding #225: the free-text question path is pure too.
        "free_text_submission",
        "chat_input_model",
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


def _app_tree() -> ast.Module:
    return ast.parse((UI_DIR / "app.py").read_text(encoding="utf-8"))


def _referenced_names(node: ast.AST) -> set[str]:
    """Every Name/Attribute identifier referenced under ``node``."""
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            names.add(child.id)
        elif isinstance(child, ast.Attribute):
            names.add(child.attr)
    return names


class TestShellTransportFailureGuard:
    """Review finding #224 RED — the shell is guarded, not decision-making.

    ``_render_chat`` consumes a network stream with no exception handling:
    a routine 429 or an api restart renders a raw traceback on the public
    page (and skips the ADR-018 footer). The shell must wrap its stream
    consumption in try/except whose handler routes through the
    presenter-exported transport-failure path — the DECISION stays pure
    (transport_failure_view / answer_status_lines); the shell only guards.
    """

    def test_chat_render_wraps_stream_consumption_in_a_transport_guard(self) -> None:
        tree = _app_tree()
        chat_renderers = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and "chat" in node.name
        ]
        assert chat_renderers, "ui/app.py must keep a chat-rendering function"
        guarded = False
        for function in chat_renderers:
            for node in ast.walk(function):
                if not isinstance(node, ast.Try):
                    continue
                handler_names: set[str] = set()
                for handler in node.handlers:
                    handler_names |= _referenced_names(handler)
                if "transport_failure_view" in handler_names:
                    guarded = True
        assert guarded, (
            "the chat renderer must wrap stream consumption in try/except "
            "whose handler folds the teed partial events through the "
            "presenter-exported transport_failure_view (finding #224)"
        )

    def test_shell_catches_the_seam_error_type_not_httpx(self) -> None:
        """The seam translates httpx failures into TransportError
        (ui.sse_client); the shell catches THAT — it never imports or
        names httpx specifics."""
        assert "httpx" not in imported_names(UI_DIR / "app.py"), (
            "ui/app.py must not import httpx; transport failures reach the "
            "shell as ui.sse_client.TransportError through the seam"
        )
        tree = _app_tree()
        caught: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                for handler in node.handlers:
                    if handler.type is not None:
                        caught |= _referenced_names(handler.type)
        assert "TransportError" in caught, (
            "the shell must catch the seam's TransportError (finding #224)"
        )

    def test_shell_renders_the_status_lines_unconditionally(self) -> None:
        """complete=False must be VISIBLE: the shell renders whatever
        answer_status_lines returns on every answer path."""
        assert "answer_status_lines" in _referenced_names(_app_tree()), (
            "ui/app.py must render answer_status_lines(view) so an "
            "incomplete stream is never presented complete-looking"
        )


class TestShellFreeTextInput:
    """Review finding #225 RED — the shell wires a real free-text input.

    The pure submission model is pinned in test_ui_starter.py; this
    structural guard pins that ui/app.py actually renders an input widget
    and routes it through the SAME pure path the starter buttons use —
    "Ask anything" must be typeable, with no shell-side query rewriting.
    """

    def test_shell_renders_a_chat_input_widget(self) -> None:
        referenced = _referenced_names(_app_tree())
        assert "chat_input" in referenced, (
            "ui/app.py must render st.chat_input — the §7.1 tagline promises "
            "'Ask anything' and the only affordance today is 13 starter buttons"
        )

    def test_shell_routes_typed_questions_through_the_pure_submission(self) -> None:
        referenced = _referenced_names(_app_tree())
        assert "free_text_submission" in referenced, (
            "typed questions must route through the presenter-exported "
            "free_text_submission (verbatim pass-through, blank rejection) — "
            "never ad-hoc shell string handling"
        )
        assert "chat_input_model" in referenced, (
            "the input's placeholder/disclosure must come from the pure "
            "chat_input_model — the logging disclosure is shown AT the input"
        )
