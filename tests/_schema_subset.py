"""Shared structured-outputs JSON-Schema subset lint (findings #203/#209/#262).

The recursive walker the #203 red phase introduced for the citation
validator's request schema, promoted to a shared helper so EVERY
structured-request builder in the repo is held to the same rule
(review finding #209): a schema sent on the ``ProviderAdapter.structured``
channel must stay inside the structured-outputs supported JSON-Schema
subset, or every live call risks a 400 (a permanent
``ProviderTransportError`` while the unit suite stays green) or a
silently stripped constraint.

Documented-unsupported keys: ``maxItems`` (not supported at all),
``minItems`` above 1, ``minimum``/``maximum``/``multipleOf``,
``minLength``/``maxLength``. And every object node must be closed — an
explicit ``additionalProperties: False`` and a ``required`` list — so
constrained decoding stays deterministic.

Any counting/shape invariant such keys used to carry must be re-homed
into prompt text ("exactly N …" where applicable) plus parser-side
validation — the ratified #203 remedy, applied to the chart planner by
finding #209.

Complexity budget (finding #262)
--------------------------------

Staying inside the supported *vocabulary* is necessary but not
sufficient: the live API also rejects schemas that exceed an
undocumented complexity limit (``400 invalid_request_error: "Schema is
too complex"`` — drawn by the chart planner's request schema during the
#162 recording session, while the citation validator's schema was
accepted the same session). :func:`assert_schema_within_complexity_budget`
bounds the measurable complexity axes so every registered structured
builder stays comfortably on the accepted side.

The budget values are EMPIRICAL, chosen offline from the one
live-rejected schema vs the two live-accepted ones (2026-09-03
measurement, compact-serialized):

===================  ==========  ================  ===========
axis                 planner     validator (45p)   classifier
                     (rejected)  (accepted)        (accepted)
===================  ==========  ================  ===========
bytes                3597        284               571
mapping nodes        88          7                 12
object nodes         15          2                 1
property keys        57          3                 6
max depth            13          7                 7
enum members         16          0                 13
if/then blocks       2           0                 0
===================  ==========  ================  ===========

Each bound sits well above every live-accepted schema and below the
rejected one; the green-phase live probe (#262 step 4) is the
confirmation that the budget lies under the real limit — tighten these
constants if that probe still draws the 400, never loosen them past a
live-rejected shape.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

#: Schema keys the structured-outputs channel does not support at all.
BANNED_KEYS = frozenset({"maxItems", "minimum", "maximum", "multipleOf", "minLength", "maxLength"})

#: Conditional/combinator keys OUTSIDE the documented supported subset
#: (finding #262). The structured-outputs docs list ``enum``/``const``/
#: ``anyOf``/``allOf``/``$ref`` as supported; ``if``/``then``/``else``/
#: ``not``/``oneOf`` are absent from that list and have never been
#: verified live (the planner schema — the only user — was rejected).
#: The conditional-required invariants they carried are re-homed to the
#: parse path (``charts.planner._parse_planner_outcome`` already
#: enforces outcome→spec / outcome→requested_data requiredness).
BANNED_CONDITIONAL_KEYS = frozenset({"if", "then", "else", "not", "oneOf"})

#: Complexity budget (finding #262) — see the module docstring table for
#: the empirical derivation. FLAGGED: these are offline estimates; the
#: green-phase live probe confirms they sit under the real API limit.
MAX_SCHEMA_BYTES = 2048  # compact json.dumps, UTF-8
MAX_SCHEMA_NODES = 48  # mapping nodes anywhere in the schema tree
MAX_OBJECT_NODES = 10  # nodes with "type": "object"
MAX_PROPERTY_KEYS = 32  # total keys across every "properties" map
MAX_NESTING_DEPTH = 10  # container depth, root = 1
MAX_ENUM_MEMBERS = 24  # total enum members across the schema


def measure_schema_complexity(schema) -> dict[str, int]:
    """The complexity metrics the #262 budget bounds, for one schema.

    Depth counts every container level (mapping or list), root = 1 —
    the same measurement the budget constants were derived with.
    """
    metrics = {
        "bytes": len(json.dumps(schema, separators=(",", ":"), ensure_ascii=False).encode()),
        "nodes": 0,
        "object_nodes": 0,
        "property_keys": 0,
        "max_depth": 0,
        "enum_members": 0,
    }

    def walk(node, depth):
        metrics["max_depth"] = max(metrics["max_depth"], depth)
        if isinstance(node, Mapping):
            metrics["nodes"] += 1
            if node.get("type") == "object":
                metrics["object_nodes"] += 1
            properties = node.get("properties")
            if isinstance(properties, Mapping):
                metrics["property_keys"] += len(properties)
            enum = node.get("enum")
            if isinstance(enum, list):
                metrics["enum_members"] += len(enum)
            for value in node.values():
                walk(value, depth + 1)
        elif isinstance(node, list):
            for value in node:
                walk(value, depth + 1)

    walk(schema, 1)
    return metrics


def assert_schema_within_complexity_budget(schema, name: str = "$") -> None:
    """Raise ``AssertionError`` when ``schema`` exceeds the #262 budget.

    Complements :func:`assert_schema_within_structured_outputs_subset`:
    the subset lint keeps the schema inside the supported *vocabulary*;
    this keeps it under the live API's complexity limit ("Schema is too
    complex", the unbilled 400 that blocked the #162 planner recording).
    ``name`` prefixes every reported message.
    """

    def check_conditionals(node, path):
        if isinstance(node, Mapping):
            for banned in sorted(BANNED_CONDITIONAL_KEYS & set(node)):
                raise AssertionError(
                    f"{name}: schema node at {path} carries {banned!r} — outside the "
                    "documented structured-outputs supported subset and never "
                    "verified live; re-home conditional requireds to the parse "
                    "path (finding #262)"
                )
            for key, value in node.items():
                check_conditionals(value, f"{path}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                check_conditionals(value, f"{path}[{index}]")

    check_conditionals(schema, "$")

    metrics = measure_schema_complexity(schema)
    budget = {
        "bytes": MAX_SCHEMA_BYTES,
        "nodes": MAX_SCHEMA_NODES,
        "object_nodes": MAX_OBJECT_NODES,
        "property_keys": MAX_PROPERTY_KEYS,
        "max_depth": MAX_NESTING_DEPTH,
        "enum_members": MAX_ENUM_MEMBERS,
    }
    for axis, ceiling in budget.items():
        assert metrics[axis] <= ceiling, (
            f"{name}: schema {axis} is {metrics[axis]}, over the #262 complexity "
            f"budget of {ceiling} — the live structured-outputs API rejected a "
            "schema of this complexity with 400 'Schema is too complex' (the "
            "#162 planner recording session); shrink the request-side schema "
            "and re-home the dropped enforcement to prompt text + validate_spec "
            f"(full metrics: {metrics})"
        )


def assert_schema_within_structured_outputs_subset(schema, name: str = "$") -> None:
    """Walk ``schema`` recursively; raise ``AssertionError`` naming the
    first node that leaves the structured-outputs supported subset.

    ``name`` prefixes every reported path (use the builder's name so a
    failure in a parameterised sweep is attributable at a glance).
    """

    def walk(node, path):
        if isinstance(node, Mapping):
            for banned in BANNED_KEYS & set(node):
                raise AssertionError(
                    f"schema node at {path} carries {banned!r}: outside the "
                    "structured-outputs supported subset (findings #203/#209)"
                )
            if "minItems" in node:
                assert node["minItems"] in (0, 1), (
                    f"schema node at {path} carries minItems={node['minItems']}: "
                    "the structured-outputs subset supports minItems only for "
                    "0 and 1 (findings #203/#209)"
                )
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False, (
                    f"object node at {path} must close with additionalProperties: False"
                )
                assert isinstance(node.get("required"), list), (
                    f"object node at {path} must carry a required list"
                )
            for key, value in node.items():
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")

    walk(schema, name)
