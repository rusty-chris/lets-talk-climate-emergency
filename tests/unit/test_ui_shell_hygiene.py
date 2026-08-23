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
        # Review finding #226: the replay-vs-stream rerun decision.
        "resolve_exchange",
        # Review finding #228: the transparency-route link derivation.
        "footer_link_line",
        # Review finding #232: calibrated-term markup + likelihood legend.
        "annotate_calibrated_terms",
        "likelihood_legend",
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

    def test_ui_vocabulary_is_set_equal_to_the_service_vocabulary(self) -> None:
        """Review finding #230 RED — parity must be BIDIRECTIONAL.

        The pairwise pins above cannot catch a service-side event
        ADDITION (exactly what open finding #220's sources event will
        be): every pair still matches while the new surface is silently
        dropped by the fold's runtime forward-compat rule. The service
        must declare its complete emit-able vocabulary
        (service.app.SSE_EVENT_NAMES) and the UI's handled-plus-ignored
        set must equal it exactly — an addition fails HERE until the UI
        decides handle-or-ignore, and an ignored event is a recorded
        decision, not an accident."""
        import service.app as service_app
        import ui.render_model as render_model

        assert hasattr(service_app, "SSE_EVENT_NAMES"), (
            "service.app must export SSE_EVENT_NAMES: the complete frozenset "
            "of emit-able SSE event names (finding #230)"
        )
        service_vocabulary = service_app.SSE_EVENT_NAMES
        ui_vocabulary = render_model.HANDLED_EVENTS | render_model.IGNORED_EVENTS

        assert ui_vocabulary == service_vocabulary, (
            f"wire-vocabulary drift: UI handles/ignores {sorted(ui_vocabulary)} "
            f"but the service declares {sorted(service_vocabulary)} — a service "
            "addition must be handled or explicitly recorded in IGNORED_EVENTS"
        )
        # Handled and ignored are disjoint: an event is one or the other.
        assert not (render_model.HANDLED_EVENTS & render_model.IGNORED_EVENTS)

    def test_generation_stream_event_names_match(self) -> None:
        """Review finding #230 RED — the #12 event names are pinned to
        NOTHING: rag/generation.py emits {"event": "text"} etc. as string
        literals, so the five UI constants are compared only against the
        test fixtures' own duplicated literals. The producer must export
        named constants and the UI pins equal to THEM."""
        import rag.generation as generation
        import ui.render_model as render_model

        for constant in (
            "TEXT_EVENT",
            "CITATION_EVENT",
            "USAGE_EVENT",
            "FOOTER_EVENT",
            "ERROR_EVENT",
        ):
            assert hasattr(generation, constant), (
                f"rag.generation must export {constant} — the #12 stream's "
                "event names are currently unpinned string literals "
                "(finding #230)"
            )
            assert getattr(render_model, constant) == getattr(generation, constant)


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


class TestShellExchangeReplayGuard:
    """Review finding #226 RED — the transport opens only on the stream branch.

    The pure replay-vs-stream decision is pinned in
    test_ui_render_model.py::TestExchangeReplay; this structural guard
    pins that the shell consults it: ui/app.py references
    resolve_exchange, and every http_chat_transport construction sits
    inside a conditional — never unconditionally on the rerun path.
    """

    def test_shell_consults_the_pure_replay_decision(self) -> None:
        assert "resolve_exchange" in _referenced_names(_app_tree()), (
            "ui/app.py must route reruns through the presenter-exported "
            "resolve_exchange decision instead of re-opening POST /chat on "
            "every script execution (finding #226)"
        )

    def test_transport_is_only_constructed_conditionally(self) -> None:
        tree = _app_tree()

        def contains_transport_call(node: ast.AST) -> bool:
            return any(
                isinstance(child, ast.Name) and child.id == "http_chat_transport"
                for child in ast.walk(node)
            )

        transport_uses = [node for node in ast.walk(tree) if contains_transport_call(node)]
        assert transport_uses, "the shell still needs the transport for fresh questions"

        # Walk with parent links: every http_chat_transport reference must
        # have an ast.If among its ancestors (the stream branch of the
        # replay decision) — an unconditional construction re-POSTs on
        # every Streamlit rerun.
        parents: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "http_chat_transport":
                ancestors = []
                current = node
                while current in parents:
                    current = parents[current]
                    ancestors.append(current)
                assert any(isinstance(ancestor, ast.If) for ancestor in ancestors), (
                    "http_chat_transport is reached unconditionally: the shell "
                    "must only construct the transport on the stream branch of "
                    "the resolve_exchange decision (finding #226)"
                )


class TestShellFooterLinks:
    """Review finding #228 RED — the shell renders the pure link line.

    The transparency routes resolve on the api/site origin, not the
    Streamlit origin, and captions don't render markdown links reliably:
    the shell must render footer_link_line(footer, base_url) via
    st.markdown.
    """

    def test_shell_renders_the_footer_link_line(self) -> None:
        referenced = _referenced_names(_app_tree())
        assert "footer_link_line" in referenced, (
            "ui/app.py must render the presenter-exported footer_link_line "
            "against SITE_URL/API_URL — the routes are dead text today "
            "(finding #228)"
        )


class TestShellRendersTheFoldedChart:
    """Review finding #229 RED — one chart rule, and it lives in the fold.

    ui/app.py re-derives the inline chart from the raw event list with
    next(... e.get("event") == "chart" ...): the fold keeps the LAST
    chart event, the shell renders the FIRST — the tested model and the
    rendered page disagree. The shell must render view.chart and nothing
    else; the fold takes chart_base_url so the shell has no reason to
    rebuild.
    """

    def test_shell_never_calls_chart_view_from_event(self) -> None:
        assert "chart_view_from_event" not in _referenced_names(_app_tree()), (
            "ui/app.py must render view.chart from the fold; re-deriving the "
            "chart from raw events diverges from the tested model "
            "(first-vs-last winner, finding #229)"
        )


class TestShellCalibratedTermSurface:
    """Review finding #232 RED — the caption's claim must be true.

    The shell currently renders a caption asserting phrases are
    highlighted while discarding the computed anchors: nothing is marked,
    no legend exists. The shell must render the answer body through the
    pure annotator and render the legend model — and the say-nothing
    caption goes.
    """

    def test_shell_renders_the_annotated_answer_and_the_legend(self) -> None:
        referenced = _referenced_names(_app_tree())
        assert "annotate_calibrated_terms" in referenced, (
            "ui/app.py must render the answer body through annotate_calibrated_terms (finding #232)"
        )
        assert "likelihood_legend" in referenced, (
            "ui/app.py must render the likelihood legend the markers reference (finding #232)"
        )

    def test_the_false_highlight_caption_is_gone(self) -> None:
        """The rendered caption claims highlighting that does not exist —
        the inflation-screenshot risk class the project avoids. Once the
        annotation is real, the claim-only caption must be deleted."""
        tree = _app_tree()
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert "Highlighted phrases" not in node.value, (
                    "ui/app.py still renders the say-nothing 'Highlighted "
                    "phrases' caption (finding #232)"
                )


#: The wire vocabulary + view kinds the shell must reference by named
#: constant, never by string literal (review finding #233). "chart"
#: appears in both sets; exact-equality scan only, so docstrings and
#: unrelated literals never false-positive.
SHELL_FORBIDDEN_LITERALS = frozenset(
    {
        "meta",
        "answer",
        "chart",
        "text",
        "citation",
        "usage",
        "footer",
        "error",
        "badge",
        "validation_degraded",
        "grounded",
        "refusal",
        "canned",
        "paused",
        "cached_starter",
    }
)


class TestShellLiteralHygiene:
    """Review finding #233 RED — no wire/kind literals, no dead fallback.

    ui/app.py branches on event.get("event") == "text" and
    view.kind != "grounded" with string literals while the pure core
    exports named constants for every one of them; and
    _render_sources re-derives `view.sources or source_list(view.chips)`
    although the fold ALWAYS populates view.sources — an unreachable
    fallback that re-imports a decision into the shell.
    """

    def test_shell_contains_no_sse_event_or_view_kind_string_literals(self) -> None:
        offending: list[str] = []
        for node in ast.walk(_app_tree()):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value in SHELL_FORBIDDEN_LITERALS
            ):
                offending.append(node.value)
        assert not offending, (
            f"ui/app.py hardcodes wire/kind literals {sorted(set(offending))}; "
            "use the presenter-exported *_EVENT / VIEW_KIND_* constants "
            "(finding #233)"
        )

    def test_shell_does_not_rederive_the_source_list(self) -> None:
        """fold_chat_stream always populates view.sources from the same
        chips; the shell's `or source_list(...)` fallback is dead code
        that re-imports a decision — render view.sources only."""
        assert "source_list" not in _referenced_names(_app_tree()), (
            "ui/app.py must render view.sources; the source_list fallback "
            "re-derivation is unreachable (finding #233)"
        )

    def test_facade_exports_the_event_and_kind_constants(self) -> None:
        """The shell can only avoid literals if the constants reach it
        through its one import surface."""
        import ui.presenters as presenters

        for name in (
            "META_EVENT",
            "ANSWER_EVENT",
            "CHART_EVENT",
            "TEXT_EVENT",
            "CITATION_EVENT",
            "USAGE_EVENT",
            "FOOTER_EVENT",
            "ERROR_EVENT",
            "BADGE_EVENT",
            "VALIDATION_DEGRADED_EVENT",
            "VIEW_KIND_GROUNDED",
            "VIEW_KIND_CHART",
            "VIEW_KIND_REFUSAL",
            "VIEW_KIND_CANNED",
            "VIEW_KIND_PAUSED",
            "VIEW_KIND_CACHED_STARTER",
        ):
            assert hasattr(presenters, name), f"ui.presenters does not export {name}"
