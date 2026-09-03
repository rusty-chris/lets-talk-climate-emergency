"""Parse backends as an optional dependency extra (issue #125, re-scoped).

The re-scope comment on #125 (ratified via the #22 red phase): the
"no torch in the api image" half is withdrawn — retrieval runs in-process
in the api, so torch is a legitimate serving dependency. What remains is
that the INGESTION parse backends (docling — which itself pulls
torch/transformers at multi-GB scale — and pymupdf) are an *offline
pipeline* concern and never belong in the serving image. This file pins
the whole contract, red-first:

1. ``pyproject.toml`` declares docling/pymupdf in the
   ``[project.optional-dependencies]`` group named ``parse`` — NOT in the
   base ``dependencies`` list, so a plain install (and therefore the api
   image's ``uv sync``) does not carry them.
2. The licensing-gate import path (``ingestion.manifest``,
   ``ingestion.gate``, ``ingestion.blocks``) and ``ingestion.parse``
   itself never import docling/fitz as a side effect — same fresh-
   interpreter probe pattern as tests/unit/test_service_import_hygiene.py.
3. When a parse backend is *invoked* without the extra installed,
   ``ingestion.parse`` raises a typed, helpful error —
   ``ParseBackendMissingError`` (a ``ModuleNotFoundError`` subclass) whose
   message names the ``parse`` extras group with an actionable install
   hint — instead of a bare ``ModuleNotFoundError`` deep inside a backend.
4. The api Docker image build does not opt back into the extra
   (Dockerfile-as-data, the #212 compose-flags convention), while the CI
   unit/integration jobs DO install it so the ingestion parse tests keep
   their runtime available.

Existing parse behaviour when the extra IS installed is deliberately not
re-pinned here: the current parse/pipeline tests are the regression net.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The agreed extras-group name (flagged decision, #125): matches the
#: module (``ingestion.parse``) and the capability being switched on.
PARSE_EXTRA = "parse"

#: The parse backend distributions that must move out of the base deps.
PARSE_BACKEND_PACKAGES = ("docling", "pymupdf")

#: Module roots that must never load on the licensing-gate import path.
#: ``fitz`` is PyMuPDF's classic import alias; ``docling_core`` ships
#: separately from ``docling`` but arrives with it.
PARSE_BACKEND_MODULES = ("docling", "docling_core", "pymupdf", "fitz")


def _pyproject() -> dict:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _requirement_name(requirement: str) -> str:
    """The distribution name of a PEP 508 requirement string, normalised."""
    for sep in ("[", ">", "<", "=", "!", "~", ";", " "):
        requirement = requirement.split(sep, 1)[0]
    return requirement.strip().lower().replace("_", "-")


# --------------------------------------------------------------------------- #
# 1. pyproject contract: base deps clean, optional group complete
# --------------------------------------------------------------------------- #
def test_base_dependencies_exclude_parse_backends() -> None:
    """Neither docling nor pymupdf appears in [project].dependencies.

    Base deps are what the api image's ``uv sync --no-dev`` installs;
    docling alone drags torch/transformers layers the serving image would
    otherwise carry for a code path it never runs (#125).
    """
    base = [_requirement_name(r) for r in _pyproject()["project"]["dependencies"]]
    offending = [name for name in PARSE_BACKEND_PACKAGES if name in base]
    assert offending == [], (
        f"parse backends {offending} are still base dependencies; they belong in "
        f"[project.optional-dependencies].{PARSE_EXTRA} (#125)"
    )


def test_optional_parse_extra_declares_both_backends() -> None:
    """[project.optional-dependencies].parse exists and pins both backends.

    The extra is the one switch dev/CI/pipeline installs flip on; losing
    either backend from it would silently break the Docling-primary /
    PyMuPDF-loud-fallback seam (DESIGN §2.4).
    """
    optional = _pyproject()["project"].get("optional-dependencies", {})
    assert PARSE_EXTRA in optional, (
        f"pyproject has no [project.optional-dependencies].{PARSE_EXTRA} group; "
        f"available groups: {sorted(optional)}"
    )
    declared = [_requirement_name(r) for r in optional[PARSE_EXTRA]]
    missing = [name for name in PARSE_BACKEND_PACKAGES if name not in declared]
    assert missing == [], (
        f"the {PARSE_EXTRA!r} extra is missing {missing}; it declares only {declared}"
    )


# --------------------------------------------------------------------------- #
# 2. Import hygiene: the licensing-gate path never loads a parse backend
# --------------------------------------------------------------------------- #
_GATE_PATH_PROBE = """
import json, sys
import ingestion
import ingestion.manifest
import ingestion.gate
import ingestion.blocks
import ingestion.parse
loaded = [name for name in {modules!r} if name in sys.modules]
print(json.dumps(loaded))
"""


def test_gate_path_import_never_loads_parse_backends() -> None:
    """Importing manifest/gate/blocks (and parse itself) pulls no backend.

    The licensing gate must run — including on machines and images without
    the ``parse`` extra — on pure imports. Fresh-interpreter probe, the
    test_service_import_hygiene.py pattern.
    """
    result = subprocess.run(
        [sys.executable, "-c", _GATE_PATH_PROBE.format(modules=PARSE_BACKEND_MODULES)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    loaded = json.loads(result.stdout.strip().splitlines()[-1])
    assert loaded == [], (
        f"importing the ingestion gate path loaded parse backends {loaded}: "
        "docling/pymupdf imports must stay lazy inside the backend functions (#125)"
    )


# --------------------------------------------------------------------------- #
# 3. Invoking a backend without the extra: typed, helpful error
# --------------------------------------------------------------------------- #
# Simulates "extra not installed" in a fresh interpreter by blocking the
# backend module roots at the import-system level, so the probe is valid
# whether or not the extra happens to be installed where tests run.
_MISSING_BACKEND_PROBE = """
import importlib.abc
import json
import sys

BLOCKED = {blocked!r}


class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError(
                "No module named " + repr(fullname), name=fullname
            )
        return None


sys.meta_path.insert(0, _Blocker())

import ingestion.parse as parse  # must import fine without the backends

error_type = getattr(parse, "ParseBackendMissingError", None)
results = {{
    "has_error_type": error_type is not None,
    "subclasses_module_not_found": bool(
        error_type is not None and issubclass(error_type, ModuleNotFoundError)
    ),
}}

calls = {{
    "docling": lambda: parse.parse_with_docling("nonexistent.pdf", "doc-1"),
    "pymupdf": lambda: parse.parse_with_pymupdf("nonexistent.pdf", "doc-1"),
    "interface": lambda: parse.parse_document("nonexistent.pdf", "doc-1"),
}}
for label, call in calls.items():
    try:
        call()
    except Exception as exc:  # noqa: BLE001 - probe records whatever raised
        results[label] = {{
            "raised": type(exc).__name__,
            "typed": bool(error_type is not None and isinstance(exc, error_type)),
            "message": str(exc),
        }}
    else:
        results[label] = {{"raised": None, "typed": False, "message": ""}}
print(json.dumps(results))
"""


def _missing_backend_probe_results() -> dict:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _MISSING_BACKEND_PROBE.format(blocked=PARSE_BACKEND_MODULES),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_missing_backend_raises_typed_error() -> None:
    """Every parse entry point raises ParseBackendMissingError when the
    extra is absent — a bare backend-internal ModuleNotFoundError is the
    unhelpful failure this replaces.

    ``ParseBackendMissingError`` subclasses ``ModuleNotFoundError`` so any
    existing ``except ImportError`` handling still works.
    """
    results = _missing_backend_probe_results()
    assert results["has_error_type"], "ingestion.parse has no ParseBackendMissingError type (#125)"
    assert results["subclasses_module_not_found"], (
        "ParseBackendMissingError must subclass ModuleNotFoundError"
    )
    untyped = {
        label: results[label]["raised"]
        for label in ("docling", "pymupdf", "interface")
        if not results[label]["typed"]
    }
    assert untyped == {}, (
        f"parse entry points raised untyped errors with the {PARSE_EXTRA!r} extra "
        f"missing: {untyped} (expected ParseBackendMissingError)"
    )


def test_missing_backend_error_names_the_extras_group() -> None:
    """The error message is actionable: it names the ``parse`` extra with
    an install hint (``--extra parse`` uv form or ``[parse]`` pip form)."""
    results = _missing_backend_probe_results()
    unhelpful = {}
    for label in ("docling", "pymupdf", "interface"):
        message = results[label]["message"]
        if f"--extra {PARSE_EXTRA}" not in message and f"[{PARSE_EXTRA}]" not in message:
            unhelpful[label] = message
    assert unhelpful == {}, (
        f"missing-backend errors do not name the {PARSE_EXTRA!r} extras group "
        f"with an install hint: {unhelpful}"
    )


# --------------------------------------------------------------------------- #
# 4. Image vs CI installs (config-as-data, the #212 convention)
# --------------------------------------------------------------------------- #
def _dockerfile_sync_lines() -> list[str]:
    text = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    return [line for line in text.splitlines() if "uv sync" in line or "pip install" in line]


def test_api_image_does_not_install_the_parse_extra() -> None:
    """No install line in the Dockerfile opts back into the parse extra.

    With the backends out of the base deps (pinned above), the image is
    clean *only* while its ``uv sync`` lines request neither
    ``--extra parse`` nor ``--all-extras`` — and no ad-hoc pip line
    reinstalls a backend.
    """
    lines = _dockerfile_sync_lines()
    assert any("uv sync" in line for line in lines), (
        "Dockerfile no longer installs via `uv sync` — update this pin to cover "
        "whatever install mechanism replaced it"
    )
    offending = [
        line
        for line in lines
        if f"--extra {PARSE_EXTRA}" in line
        or "--all-extras" in line
        or any(pkg in line for pkg in PARSE_BACKEND_PACKAGES)
    ]
    assert offending == [], f"the api image install requests the parse backends: {offending} (#125)"


def test_ci_test_jobs_install_the_parse_extra() -> None:
    """The CI unit and integration jobs sync WITH the parse extra.

    The ingestion parse seam's runtime must stay installed where the test
    suite runs, so today's (and future real-backend) parse tests keep
    running once the backends leave the base dependencies.
    """
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    problems: list[str] = []
    for job_name in ("unit", "integration"):
        steps = workflow["jobs"][job_name]["steps"]
        sync_runs = [
            step["run"]
            for step in steps
            if isinstance(step.get("run"), str) and "uv sync" in step["run"]
        ]
        if not sync_runs:
            problems.append(f"{job_name}: no `uv sync` step found")
            continue
        if not any(f"--extra {PARSE_EXTRA}" in run or "--all-extras" in run for run in sync_runs):
            problems.append(f"{job_name}: uv sync without the {PARSE_EXTRA!r} extra: {sync_runs}")
    assert problems == [], (
        f"CI jobs would run the suite without the parse backends installed: {problems}"
    )
