"""Review finding #230 RED — the service declares its complete SSE vocabulary.

The wire-vocabulary parity guard (test_ui_shell_hygiene.py) can only be
bidirectional if the SERVICE side declares every event name it can emit:
``service.app.SSE_EVENT_NAMES`` must be the complete frozenset, and every
producer module (service/app.py, rag/generation.py, rag/citation_validator.py)
must emit through NAMED constants that are members of it — a rename, typo
or unrecorded addition on the producer side fails here, not in a deployed
UI silently dropping a surface.

Enumeration is programmatic and static: every ``{"event": ...}`` dict
literal in the three producer modules is collected from the AST and its
event name resolved (constant reference -> the module's value; string
literal -> itself). The existing fake-deps unit suites cover the dynamic
routes; this pins the vocabulary declaration itself with no network and
no app construction.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The producer modules whose emitted event names make up the wire.
PRODUCER_MODULES = (
    "service/app.py",
    "rag/generation.py",
    "rag/citation_validator.py",
)


def _emitted_event_values(path: Path) -> list[ast.expr]:
    """Every value expression of an ``"event"`` key in a dict literal."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values: list[ast.expr] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values, strict=True):
            if isinstance(key, ast.Constant) and key.value == "event":
                values.append(value)
    return values


def _resolve_event_name(value: ast.expr, module: object) -> str:
    """A string literal resolves to itself; a Name to the module constant."""
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return value.value
    if isinstance(value, ast.Name):
        resolved = getattr(module, value.id, None)
        assert isinstance(resolved, str), (
            f"event-name constant {value.id!r} did not resolve to a string"
        )
        return resolved
    raise AssertionError(f"unresolvable event-name expression: {ast.dump(value)}")


def test_service_declares_the_complete_sse_vocabulary() -> None:
    """``service.app.SSE_EVENT_NAMES`` exists and covers the whole wire —
    the #22 meta/answer/chart names, the #12 generation-stream names, and
    the #13 validator names."""
    import service.app as service_app

    assert hasattr(service_app, "SSE_EVENT_NAMES"), (
        "service.app must export SSE_EVENT_NAMES: frozenset of every "
        "emit-able SSE event name (finding #230)"
    )
    vocabulary = service_app.SSE_EVENT_NAMES
    assert isinstance(vocabulary, frozenset)
    assert vocabulary == frozenset(
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
        }
    )


@pytest.mark.parametrize("module_path", PRODUCER_MODULES)
def test_every_emitted_event_name_is_in_the_declared_vocabulary(module_path: str) -> None:
    """Statically enumerate every ``{"event": ...}`` emission in the
    producer modules and assert each resolves into SSE_EVENT_NAMES — an
    event the service can emit but did not declare is exactly the drift
    the bidirectional parity guard exists to catch."""
    import importlib

    import service.app as service_app

    module = importlib.import_module(module_path.removesuffix(".py").replace("/", "."))
    values = _emitted_event_values(REPO_ROOT / module_path)
    assert values, f"{module_path} is expected to emit SSE events"

    assert hasattr(service_app, "SSE_EVENT_NAMES"), (
        "service.app must export SSE_EVENT_NAMES (finding #230)"
    )
    for value in values:
        name = _resolve_event_name(value, module)
        assert name in service_app.SSE_EVENT_NAMES, (
            f"{module_path} emits event {name!r} which is not in the "
            "declared SSE_EVENT_NAMES vocabulary"
        )


@pytest.mark.parametrize("module_path", PRODUCER_MODULES)
def test_producers_emit_through_named_constants_not_literals(module_path: str) -> None:
    """The #12 names are pinned to nothing today because rag/generation.py
    emits {"event": "text"} as a string literal: parity against literals
    is duplication agreeing with duplication. Every emission must
    reference a NAMED constant so the pairwise pins bind the producer."""
    for value in _emitted_event_values(REPO_ROOT / module_path):
        assert not isinstance(value, ast.Constant), (
            f"{module_path} emits an SSE event with the string literal "
            f"{value.value!r}; emit through a named module constant so the "
            "vocabulary pins bind the producer (finding #230)"
        )
