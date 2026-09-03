"""Issue #220 RED — the UI folds the ``sources`` event into the §3.6 panel.

The service half (tests/unit/test_service_sources_event.py) pins the wire:
one ``sources`` event per grounded exchange, after ``meta``, before the
first ``text``, every excerpt licence-bounded (fail-closed to ``None``).
This suite pins the UI half:

- Vocabulary: ``ui.render_model.SOURCES_EVENT`` equals the service's and
  is HANDLED (the #230 bidirectional parity guard holds with the new
  event on both sides — extended coherently, not tripped).
- :func:`ui.render_model.build_sources_panel`: the pure wire-entries ->
  :class:`SourcesPanel` grouping — voices passages separated under the
  §2.1/§7.2 "About the movement" styling with their own attribution,
  everything else (including forward-compat unknown source_types, which
  must never masquerade as movement testimony) under evidence.
- The fold: ``view.sources_panel`` from the stream's single sources
  event on grounded exchanges; ``None`` with no event (an older
  service); ALWAYS ``None`` on non-grounded kinds (a protocol-breaching
  sources event on a refusal is never honoured — nothing is dressed up
  as grounding). Excerpts ride VERBATIM: ``None`` stays ``None`` (the
  licensing wall is the service's; the UI never fabricates or extends).
- Forward compatibility unchanged: genuinely unknown event names are
  still ignored at runtime.
- The facade + shell: ``ui.presenters`` exports the panel contract, and
  the Streamlit shell renders ``view.sources_panel`` with the voices
  heading from the pure constant (structural pins, same style as the
  #224/#226 guards).
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from ui.render_model import (
    EVIDENCE_PANEL_HEADING,
    HANDLED_EVENTS,
    IGNORED_EVENTS,
    SOURCES_EVENT,
    VIEW_KIND_GROUNDED,
    VOICES_PANEL_HEADING,
    SourcePanelEntry,
    SourcesPanel,
    build_sources_panel,
    fold_chat_stream,
)

UI_DIR = Path(__file__).resolve().parents[2] / "ui"


def wire_entry(index: int, *, source_type: str = "evidence", **overrides: Any) -> dict[str, Any]:
    """One sources-event wire entry, exactly the service's closed shape."""
    entry: dict[str, Any] = {
        "doc_id": f"syn-doc-{index}",
        "chunk_id": f"syn-doc-{index}::c{index:04d}",
        "title": f"Invented Source Title {index}",
        "attribution_text": f"Invented Attribution {index}",
        "canonical_url": f"https://example.invalid/doc/{index}",
        "source_type": source_type,
        "source_tier": "A",
        "permitted_context": "open",
        "excerpt": f"Invented bounded excerpt {index}.",
        "excerpt_truncated": False,
    }
    entry.update(overrides)
    return entry


def meta_event(mode: str = "live") -> dict[str, Any]:
    return {
        "event": "meta",
        "data": {"disclosure": "One-line logging disclosure.", "preamble_note": None, "mode": mode},
    }


def sources_event(*entries: dict[str, Any]) -> dict[str, Any]:
    return {"event": SOURCES_EVENT, "data": {"sources": list(entries)}}


class TestVocabularyExtension:
    def test_sources_event_name_matches_the_service(self) -> None:
        import service.app as service_app

        assert SOURCES_EVENT == service_app.SOURCES_EVENT

    def test_sources_is_handled_not_ignored(self) -> None:
        """#230 extended coherently: the new event is a HANDLED surface,
        so the bidirectional set-equality guard passes with the event on
        both sides — the guard tripping was the design working; this is
        the recorded handle decision."""
        assert SOURCES_EVENT in HANDLED_EVENTS
        assert SOURCES_EVENT not in IGNORED_EVENTS

    def test_voices_heading_is_the_design_wording(self) -> None:
        assert VOICES_PANEL_HEADING == "About the movement", (
            "§7.2 pins the voices styling wording; the shell renders this "
            "constant, never its own literal"
        )
        assert EVIDENCE_PANEL_HEADING


class TestBuildSourcesPanel:
    def test_groups_by_source_type_preserving_wire_order(self) -> None:
        entries = [
            wire_entry(0),
            wire_entry(1, source_type="voices"),
            wire_entry(2),
            wire_entry(3, source_type="voices"),
        ]
        panel = build_sources_panel(entries)
        assert isinstance(panel, SourcesPanel)
        assert [e.chunk_id for e in panel.evidence] == [
            entries[0]["chunk_id"],
            entries[2]["chunk_id"],
        ]
        assert [e.chunk_id for e in panel.voices] == [
            entries[1]["chunk_id"],
            entries[3]["chunk_id"],
        ]

    def test_carries_every_wire_field_verbatim(self) -> None:
        entry = wire_entry(
            0,
            source_tier="B",
            permitted_context="non-commercial-educational",
            excerpt="A tight bounded excerpt.",
            excerpt_truncated=True,
        )
        (view_entry,) = build_sources_panel([entry]).evidence
        assert isinstance(view_entry, SourcePanelEntry)
        assert view_entry.doc_id == entry["doc_id"]
        assert view_entry.chunk_id == entry["chunk_id"]
        assert view_entry.title == entry["title"]
        assert view_entry.attribution == entry["attribution_text"]
        assert view_entry.canonical_url == entry["canonical_url"]
        assert view_entry.source_type == "evidence"
        assert view_entry.source_tier == "B"
        assert view_entry.permitted_context == "non-commercial-educational"
        assert view_entry.excerpt == "A tight bounded excerpt."
        assert view_entry.excerpt_truncated is True

    def test_voices_entries_carry_their_own_attribution(self) -> None:
        """§2.1: voices passages render under 'About the movement' with
        the movement source's OWN attribution text — never inherited from
        an evidence sibling."""
        entries = [
            wire_entry(0, attribution_text="An Invented Assessment Body"),
            wire_entry(
                1,
                source_type="voices",
                attribution_text="An Invented Movement Voice (first-party)",
            ),
        ]
        panel = build_sources_panel(entries)
        (voice,) = panel.voices
        assert voice.attribution == "An Invented Movement Voice (first-party)"
        assert voice.source_type == "voices"

    def test_none_excerpt_stays_none(self) -> None:
        """The fail-closed metadata-only state rides through untouched:
        the UI NEVER fabricates, pads or substitutes an excerpt the
        licensing wall refused."""
        entry = wire_entry(
            0,
            excerpt=None,
            excerpt_truncated=False,
            source_tier=None,
            permitted_context="mystery-licence",
        )
        (view_entry,) = build_sources_panel([entry]).evidence
        assert view_entry.excerpt is None
        assert view_entry.excerpt_truncated is False
        assert view_entry.source_tier is None

    def test_unknown_source_type_never_masquerades_as_voices(self) -> None:
        entries = [wire_entry(0, source_type="testimony-of-the-future")]
        panel = build_sources_panel(entries)
        assert panel.voices == ()
        assert len(panel.evidence) == 1

    def test_blank_title_falls_back_to_attribution_then_chunk_id(self) -> None:
        with_attr = wire_entry(0, title=None)
        bare = wire_entry(1, title=None, attribution_text=None)
        panel = build_sources_panel([with_attr, bare])
        assert panel.evidence[0].title == with_attr["attribution_text"]
        assert panel.evidence[1].title == bare["chunk_id"]

    def test_empty_wire_list_is_an_empty_panel(self) -> None:
        panel = build_sources_panel([])
        assert panel.evidence == () and panel.voices == ()


class TestFoldSourcesPanel:
    def _grounded_stream(self, *sources: dict[str, Any]) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = [meta_event()]
        if sources:
            events.append(sources_event(*sources))
        events += [
            {"event": "text", "data": {"text": "An invented grounded sentence."}},
            {"event": "footer", "data": {"text": "Synthetic footer."}},
        ]
        return events

    def test_grounded_stream_folds_the_panel(self) -> None:
        entries = (wire_entry(0), wire_entry(1, source_type="voices"))
        view = fold_chat_stream(self._grounded_stream(*entries))
        assert view.kind == VIEW_KIND_GROUNDED
        assert view.sources_panel is not None
        assert view.sources_panel == build_sources_panel(list(entries))
        # The #18 chip-derived "Sources (n)" subset is unchanged alongside.
        assert view.complete is True

    def test_stream_without_a_sources_event_folds_to_none(self) -> None:
        view = fold_chat_stream(self._grounded_stream())
        assert view.sources_panel is None, (
            "an older service sends no sources event; the UI shows no panel "
            "rather than fabricating one"
        )

    @pytest.mark.parametrize("kind", ["refusal", "canned", "paused"])
    def test_non_grounded_kinds_never_carry_a_panel(self, kind: str) -> None:
        """The service pins that these routes emit NO sources event; a
        protocol-breaching one that appears anyway is not honoured —
        exactly the existing no-chips/no-sources honesty rule."""
        events = [
            meta_event(),
            sources_event(wire_entry(0)),
            {"event": "answer", "data": {"kind": kind, "text": "A canned/refusal text."}},
        ]
        view = fold_chat_stream(events)
        assert view.kind == kind
        assert view.sources_panel is None

    def test_error_terminated_stream_keeps_the_panel(self) -> None:
        """The sources event precedes the first text event, so on a
        mid-stream error the panel HAS honestly arrived: retrieval
        metadata is kept (like the delivered text prefix), while badges
        and the footer follow the existing error rules."""
        events = [
            meta_event(),
            sources_event(wire_entry(0)),
            {"event": "text", "data": {"text": "A delivered prefix."}},
            {"event": "error", "data": {"type": "overloaded_error", "message": "upstream"}},
        ]
        view = fold_chat_stream(events)
        assert view.error is not None and view.complete is False
        assert view.sources_panel is not None
        assert [e.chunk_id for e in view.sources_panel.evidence] == [wire_entry(0)["chunk_id"]]

    def test_unknown_event_names_are_still_ignored(self) -> None:
        """Runtime forward-compat is unchanged by the new handled event."""
        events = self._grounded_stream(wire_entry(0))
        events.insert(2, {"event": "hologram", "data": {"anything": True}})
        view = fold_chat_stream(events)
        assert view.text == "An invented grounded sentence."
        assert view.sources_panel is not None


class TestPresenterFacade:
    def test_facade_exposes_the_sources_panel_contract(self) -> None:
        import ui.presenters as presenters

        for name in (
            "SOURCES_EVENT",
            "SourcePanelEntry",
            "SourcesPanel",
            "build_sources_panel",
            "VOICES_PANEL_HEADING",
            "EVIDENCE_PANEL_HEADING",
        ):
            assert hasattr(presenters, name), f"ui.presenters does not export {name}"


class TestShellRendersThePanel:
    """Structural pins (same style as the #224/#226 shell guards): the
    Streamlit shell draws what the pure panel model says — it references
    ``view.sources_panel`` and the pure voices heading constant, never a
    shell-side literal or regrouping of its own."""

    def _referenced_names(self) -> set[str]:
        tree = ast.parse((UI_DIR / "app.py").read_text(encoding="utf-8"))
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
        return names

    def test_shell_renders_the_sources_panel_view_model(self) -> None:
        assert "sources_panel" in self._referenced_names(), (
            "ui/app.py must render view.sources_panel — the §3.6 retrieved-"
            "passages panel (issue #220)"
        )

    def test_shell_uses_the_pure_voices_heading(self) -> None:
        assert "VOICES_PANEL_HEADING" in self._referenced_names(), (
            "the 'About the movement' voices styling heading is the pure "
            "core's constant; the shell renders it, never its own literal"
        )
