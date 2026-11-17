"""Code provenance for the reused, frozen Phase 3 modules (G1, condition 1).

v1 never copies or edits phase3/{harness,score,report}.py: it ships them
verbatim as top-level modules (pyproject `py-modules`) and imports them.
To make that reuse auditable, every `mpe-agent run` records the resolved
file path and sha256 of the `harness`, `score` and `report` modules it
actually imported, both in the run-level metadata and on each JSONL
record. A matching CI test asserts the installed `score.py` is
byte-identical to `phase3/score.py` (see tests/test_v1_module_provenance.py),
so a divergence between the shipped scorer and the frozen one is caught.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

# Importing the adapter runs the sys.path shim (dev) or relies on the
# installed top-level modules (after `pip install .`); either way the three
# frozen modules are importable by their flat names afterwards.
from . import openai_adapter as _oa  # noqa: F401  (import side effect: sys.path)

import harness as _hz  # noqa: E402
import report as _rpt  # noqa: E402
import score as _sc  # noqa: E402

_MODULES = {"harness": _hz, "score": _sc, "report": _rpt}


def module_provenance() -> dict[str, dict[str, str]]:
    """{module_name: {"path": <abs path>, "sha256": <hex>}} for the three
    frozen Phase 3 modules this process imported."""
    out: dict[str, dict[str, str]] = {}
    for name, module in _MODULES.items():
        path = Path(module.__file__).resolve()
        out[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return out
