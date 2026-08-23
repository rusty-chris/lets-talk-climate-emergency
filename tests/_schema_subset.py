"""Shared structured-outputs JSON-Schema subset lint (findings #203/#209).

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
"""

from __future__ import annotations

from collections.abc import Mapping

#: Schema keys the structured-outputs channel does not support at all.
BANNED_KEYS = frozenset({"maxItems", "minimum", "maximum", "multipleOf", "minLength", "maxLength"})


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
