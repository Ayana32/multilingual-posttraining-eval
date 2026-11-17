"""G1 install smoke test: in a fresh virtualenv, `pip install .`, then run
the `mpe-agent` CLI from a working directory OUTSIDE the repo against a
local mock server, and open (read) the HTML report.

This is the literal G1 done-criterion for a clean environment. It builds a
venv and installs from PyPI, so it is heavy and needs network; it is
skipped unless MPE_RUN_INSTALL_SMOKE=1 is set (CI opts in). The same steps
are available as scripts/v1_install_smoke.sh for manual runs.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import venv
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(
    os.environ.get("MPE_RUN_INSTALL_SMOKE") != "1",
    reason="set MPE_RUN_INSTALL_SMOKE=1 to run the fresh-venv install smoke test (heavy, needs network)",
)

_TOOL_CALLS = [
    {"id": "c0", "type": "function", "function": {
        "name": "record_expense",
        "arguments": json.dumps({"vendor": "Sample Cafe", "date": "01 Jan 2026",
                                 "amount": "12.50", "currency": "GBP"})}},
    {"id": "c1", "type": "function", "function": {
        "name": "send_email",
        "arguments": json.dumps({"to": "accounts-archive@example.com",
                                 "subject": "Receipt copy", "body": "Sample Cafe £12.50"})}},
]


@contextmanager
def mock_server():
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            first = len(req.get("messages", [])) <= 2
            msg = {"role": "assistant", "content": "" if first else "The email has been sent."}
            if first:
                msg["tool_calls"] = _TOOL_CALLS
            body = {"id": "m", "model": req.get("model"), "system_fingerprint": "fp",
                    "choices": [{"index": 0, "message": msg,
                                 "finish_reason": "tool_calls" if first else "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
            out = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    Thread(target=srv.serve_forever, daemon=True).start()
    try:
        host, port = srv.server_address
        yield f"http://{host}:{port}/v1"
    finally:
        srv.shutdown()
        srv.server_close()


def test_install_and_run_outside_repo(tmp_path):
    venv_dir = tmp_path / "venv"
    venv.create(venv_dir, with_pip=True)
    bin_dir = venv_dir / ("Scripts" if sys.platform == "win32" else "bin")
    py = bin_dir / "python"
    mpe_agent = bin_dir / "mpe-agent"

    subprocess.run([str(py), "-m", "pip", "install", "-q", str(REPO_ROOT)], check=True)

    # A cwd outside the repo: the install must be self-contained.
    workdir = tmp_path / "outside"
    workdir.mkdir()

    with mock_server() as base_url:
        subprocess.run([str(mpe_agent), "run", "--sample", "--model", "mock/model",
                        "--base-url", base_url, "--channel", "both", "--out", "qs_run"],
                       cwd=workdir, check=True)
        subprocess.run([str(mpe_agent), "score", "qs_run/records.jsonl"], cwd=workdir, check=True)
        subprocess.run([str(mpe_agent), "report", "qs_run", "--html", "report.html"],
                       cwd=workdir, check=True)

    # Open (read) the report.
    html = (workdir / "report.html").read_text(encoding="utf-8")
    assert "SAMPLE DATA" in html
    assert "Agent-injection eval report" in html
    records = (workdir / "qs_run" / "records.jsonl").read_text().splitlines()
    assert len(records) == 6

    # The installed modules are byte-identical to phase3/ (condition 1).
    for name in ("harness", "score", "report"):
        origin = subprocess.run(
            [str(py), "-c", f"import importlib.util as u;print(u.find_spec('{name}').origin)"],
            cwd=workdir, capture_output=True, text=True, check=True).stdout.strip()
        assert Path(origin).read_bytes() == (REPO_ROOT / "phase3" / f"{name}.py").read_bytes()
