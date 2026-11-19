"""G1 condition 1 (CI): the reused `harness`, `score` and `report`
modules must be byte-identical to the frozen phase3/ sources.

v1 ships them as top-level py-modules straight from phase3/ (pyproject
`py-modules` + package-dir), never a copy or an edit. This test resolves
each module the way the running code imports it and compares its file
bytes to phase3/<name>.py. In a clean `pip install .` the resolved origin
is the installed site-packages copy, so this is the real guard that the
shipped scorer equals the frozen one; in an in-repo/editable checkout the
origin IS phase3/<name>.py, so it trivially holds.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

import mpe_agent.openai_adapter  # noqa: F401  (runs the sys.path shim)

REPO_ROOT = Path(__file__).resolve().parent.parent
PHASE3 = REPO_ROOT / "phase3"


@pytest.mark.parametrize("module_name", ["harness", "score", "report"])
def test_installed_module_matches_phase3(module_name: str) -> None:
    spec = importlib.util.find_spec(module_name)
    assert spec is not None and spec.origin, f"{module_name} is not importable"
    installed = Path(spec.origin).read_bytes()
    frozen = (PHASE3 / f"{module_name}.py").read_bytes()
    assert installed == frozen, (
        f"{module_name} resolved to {spec.origin} differs from phase3/{module_name}.py; "
        "the reused Phase 3 module must ship byte-identical (G1 condition 1)."
    )


def test_provenance_reports_those_modules() -> None:
    from mpe_agent.provenance import module_provenance

    prov = module_provenance()
    assert set(prov) == {"harness", "score", "report"}
    for name, info in prov.items():
        assert len(info["sha256"]) == 64
        assert Path(info["path"]).name == f"{name}.py"
