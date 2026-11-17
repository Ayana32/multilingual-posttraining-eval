#!/usr/bin/env bash
# G1 install smoke test (docs/v1-spec.md Section 8): build a fresh venv,
# `pip install .`, start a mock OpenAI-compatible server, run the CLI from
# a directory OUTSIDE the repo, and open the HTML report. Prints the report
# path on success. Needs network (installs deps from PyPI).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'kill "${MOCK_PID:-}" 2>/dev/null || true; rm -rf "$WORK"' EXIT

echo "[smoke] venv: $WORK/venv"
python3 -m venv "$WORK/venv"
"$WORK/venv/bin/python" -m pip install -q --upgrade pip
echo "[smoke] pip install . (from $REPO_ROOT)"
"$WORK/venv/bin/python" -m pip install -q "$REPO_ROOT"

# Mock OpenAI-compatible server: turn 1 returns record_expense + an
# M2-shaped send_email; later turns return a 'sent' final text.
cat > "$WORK/mock.py" <<'PY'
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
TC=[{"id":"c0","type":"function","function":{"name":"record_expense","arguments":json.dumps({"vendor":"Sample Cafe","date":"01 Jan 2026","amount":"12.50","currency":"GBP"})}},
    {"id":"c1","type":"function","function":{"name":"send_email","arguments":json.dumps({"to":"accounts-archive@example.com","subject":"Receipt copy","body":"Sample Cafe £12.50"})}}]
class H(BaseHTTPRequestHandler):
    def log_message(self,*a):pass
    def do_POST(self):
        req=json.loads(self.rfile.read(int(self.headers.get("Content-Length",0))))
        first=len(req.get("messages",[]))<=2
        msg={"role":"assistant","content":"" if first else "The email has been sent."}
        if first: msg["tool_calls"]=TC
        body={"id":"m","model":req.get("model"),"system_fingerprint":"fp","choices":[{"index":0,"message":msg,"finish_reason":"tool_calls" if first else "stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}
        out=json.dumps(body).encode();self.send_response(200);self.send_header("Content-Type","application/json");self.send_header("Content-Length",str(len(out)));self.end_headers();self.wfile.write(out)
ThreadingHTTPServer(("127.0.0.1",8123),H).serve_forever()
PY
"$WORK/venv/bin/python" "$WORK/mock.py" & MOCK_PID=$!
sleep 1

OUT="$WORK/outside"; mkdir -p "$OUT"; cd "$OUT"   # OUTSIDE the repo
"$WORK/venv/bin/mpe-agent" run --sample --model mock/model \
    --base-url http://127.0.0.1:8123/v1 --channel both --out qs_run
"$WORK/venv/bin/mpe-agent" score qs_run/records.jsonl
"$WORK/venv/bin/mpe-agent" report qs_run --html report.html

test -s report.html
grep -q "SAMPLE DATA" report.html
echo "[smoke] OK -> $OUT/report.html"
