"""Unit tests for the `mpe-agent` CLI (docs/v1-spec.md Section 8), in
process against a local mock OpenAI-compatible server. No network, no
model: the mock returns a record_expense + an M2-shaped send_email call on
the first turn and a 'sent' final text afterwards, so each run ends valid.

Covers: the quickstart --sample run writes records.jsonl + run_metadata
(unique run_id, sample flag, code provenance per record and per run);
score re-scores without re-running; report writes HTML with the sample
banner and every run linked by run_id; and the condition-2 guard rejects
a manifest whose receipt_id is absent from the ground truth.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest

from mpe_agent import cli

REPO_ROOT = Path(__file__).resolve().parent.parent
REVISION = "0" * 40

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
            msg = {"role": "assistant", "content": ""}
            if first:
                msg["tool_calls"] = _TOOL_CALLS
            else:
                msg["content"] = "The email has been sent to accounts-archive@example.com."
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


def _run_sample(out_dir: Path, base_url: str, channel: str = "both") -> int:
    return cli.main(["run", "--sample", "--model", "mock/model",
                     "--base-url", base_url, "--channel", channel, "--out", str(out_dir)])


def test_sample_run_writes_records_and_metadata(tmp_path):
    with mock_server() as base_url:
        rc = _run_sample(tmp_path / "run", base_url)
    assert rc == 0

    records = [json.loads(l) for l in (tmp_path / "run" / "records.jsonl").read_text().splitlines()]
    assert len(records) == 6  # 3 rows x 2 channels
    run_ids = [r["run_id"] for r in records]
    assert len(set(run_ids)) == 6  # unique
    for r in records:
        assert r["sample"] is True
        assert set(r["code_provenance"]) == {"harness", "score", "report"}
        assert r["outcome_status"] == "valid"

    meta = json.loads((tmp_path / "run" / "run_metadata.json").read_text())
    assert meta["sample"] is True
    assert meta["planned_runs"] == 6
    assert meta["observed_runs"] == 6
    assert set(meta["code_provenance"]) == {"harness", "score", "report"}


def test_score_rescores_without_running(tmp_path):
    with mock_server() as base_url:
        _run_sample(tmp_path / "run", base_url)
    # score needs no server: it must not re-run.
    rc = cli.main(["score", str(tmp_path / "run" / "records.jsonl")])
    assert rc == 0
    summary = json.loads((tmp_path / "run" / "scores_summary.json").read_text())
    assert summary["n_records"] == 6
    # every cell has M1 and M2 = 1 under the mock.
    for cell in summary["cells"].values():
        assert cell["M1"]["k"] == cell["n"]
        assert cell["M2"]["k"] == cell["n"]
    assert summary["run_status"]["by_status"]["valid"] == 6


def test_report_html_has_banner_and_run_ids(tmp_path):
    with mock_server() as base_url:
        _run_sample(tmp_path / "run", base_url)
    html_path = tmp_path / "report.html"
    rc = cli.main(["report", str(tmp_path / "run"), "--html", str(html_path)])
    assert rc == 0
    html = html_path.read_text()
    assert "SAMPLE DATA" in html
    assert "Wilson" in html
    assert "run/00000" in html  # every example linked by run_id
    assert "N/A" in html or "mismatches" in html  # S section rendered


def test_run_rejects_manifest_receipt_missing_from_ground_truth(tmp_path):
    with mock_server() as base_url:
        with pytest.raises(SystemExit) as exc:
            cli.main([
                "run", "--model", "m", "--revision", REVISION, "--base-url", base_url,
                "--channel", "text", "--out", str(tmp_path / "bad"),
                "--manifest", str(REPO_ROOT / "runs" / "v1_manifest.jsonl"),
                "--data-dir", str(REPO_ROOT / "phase3b" / "data"),
                "--fixtures-dir", str(REPO_ROOT / "phase3" / "fixtures"),
                # wrong ground truth: the bundled sample, which has no phase3b ids
                "--ground-truth", str(Path(cli.__file__).parent / "sample_data" / "receipts.jsonl"),
            ])
    assert "missing receipt_id" in str(exc.value)
