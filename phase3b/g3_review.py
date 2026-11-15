#!/usr/bin/env python3
"""Phase 3b G3 (b)+(c) local review viewer (docs/phase3b-replication.md
Section 10; one-pass deviation logged in docs/phase3b-deviations.md 2026-11-15).

A localhost-only http.server UI that walks the 440 images cell by cell
(arm, font, variant) and records, by hand, the Section 10 (b) and (c) checks for
every image. It does not re-render anything and it never decides a check: it
only shows what G3 (a) already recorded and stores the human's entries.

  cell view  -- for one cell's 40 images, each row shows a full-resolution crop
                of that image's footer and of its TOTAL-to-footer boundary
                (crop boxes = the draw bboxes in report/phase3b/g3a_results.json
                + a small margin), the image's reference transcript footer lines,
                and the check buttons that apply to that variant (n/a cells are
                shown locked). Clicking a row opens the full image.
  body view  -- the 40 Section 10 (c) sampled images, each as a body crop beside
                that image's transcript body lines, with the body_line_by_line
                button.

Every judgment is per image and per check. A fail requires a note. There is no
bulk-pass and no automatic judgment. Progress is written to the gitignored
results/phase3b/g3_progress.json after each change and is resumable; the index
shows how many cells remain.

  python3 phase3b/g3_review.py            # serve on http://127.0.0.1:8765
  python3 phase3b/g3_review.py --export   # merge progress into the CSV

--export merges the recorded judgments into report/phase3b/g3_manual_review.csv
by image_id, refusing on any row-count / id / order change, any still-empty
required cell, or any value other than pass / fail / n/a, then prints the value
counts per column and every fail with its note.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent

G3A_PATH = REPO_ROOT / "report" / "phase3b" / "g3a_results.json"
CSV_PATH = REPO_ROOT / "report" / "phase3b" / "g3_manual_review.csv"
PROGRESS_PATH = REPO_ROOT / "results" / "phase3b" / "g3_progress.json"
IMAGES_DIR = REPO_ROOT / "phase3b" / "data" / "images"

CROP_MARGIN = 14              # px added around a crop's draw-bbox union
VALID_VALUES = ("pass", "fail", "n/a")

# Section 10 (b)+(c) checks that are reviewed from the footer / TOTAL-to-footer
# crops in the cell view, in the CSV's column order.
CELL_CHECKS = [
    "korean_renders", "receipt_copy_one_line", "address_one_line",
    "footer_not_cut_or_overlapping", "body_footer_spacing_ok", "footer_line_by_line",
]
# The one check reviewed from the body crop (the 40 sampled images).
BODY_CHECK = "body_line_by_line"


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


mr = _load("phase3b_manual_review", "phase3b/make_manual_review.py")
ALL_CHECKS = mr.CHECK_COLUMNS            # the 7 reviewer check columns
COLUMNS = mr.COLUMNS


# --------------------------------------------------------------------------- #
# image_id parsing and cell grouping
# --------------------------------------------------------------------------- #
def parse_image_id(image_id: str) -> dict:
    """A|<variant>|<rid>  or  B|<font>|<variant>|<rid>."""
    parts = image_id.split("|")
    if parts[0] == "A":
        _, variant, rid = parts
        return {"arm": "A", "font": None, "variant": variant, "receipt_id": rid}
    _, font, variant, rid = parts
    return {"arm": "B", "font": font, "variant": variant, "receipt_id": rid}


def cell_of(image_id: str) -> str:
    """The (arm, font, variant) cell key an image belongs to (its image_id
    without the receipt id)."""
    return image_id.rsplit("|", 1)[0]


def cell_order(rows: list[dict]) -> list[str]:
    """Cell keys in first-appearance (canonical) order."""
    seen: list[str] = []
    for r in rows:
        c = cell_of(r["image_id"])
        if c not in seen:
            seen.append(c)
    return seen


# --------------------------------------------------------------------------- #
# loading CSV / G3(a) / progress
# --------------------------------------------------------------------------- #
def load_rows(csv_path: Path = CSV_PATH) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_g3a(path: Path = G3A_PATH) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {img["image_id"]: img for img in data["images"]}


def load_progress(path: Path = PROGRESS_PATH) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"judgments": {}}


def save_progress(progress: dict, path: Path = PROGRESS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(progress, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                    encoding="utf-8")


# --------------------------------------------------------------------------- #
# which checks apply to a row (n/a cells are locked, never editable)
# --------------------------------------------------------------------------- #
def editable_checks(row: dict, checks: list[str]) -> list[str]:
    """Of `checks`, the ones that are not n/a for this row (i.e. the CSV left
    them empty to be filled). n/a is from the Section 10 prefill and is locked."""
    return [c for c in checks if row.get(c, "") != "n/a"]


def record_judgment(progress: dict, image_id: str, column: str, value: str, note: str) -> None:
    """Store one human entry. value must be pass/fail; a fail requires a note.
    Raises ValueError otherwise (the server surfaces it; nothing is stored)."""
    if column not in ALL_CHECKS:
        raise ValueError(f"unknown check column: {column}")
    if value not in ("pass", "fail"):
        raise ValueError(f"value must be 'pass' or 'fail', got {value!r}")
    note = (note or "").strip()
    if value == "fail" and not note:
        raise ValueError("a fail requires a note")
    progress.setdefault("judgments", {}).setdefault(image_id, {})[column] = {
        "value": value, "note": note,
    }


def judgment_for(progress: dict, image_id: str, column: str) -> dict | None:
    return progress.get("judgments", {}).get(image_id, {}).get(column)


# --------------------------------------------------------------------------- #
# completion / remaining-cells counter
# --------------------------------------------------------------------------- #
def _row_checks_done(row: dict, progress: dict, checks: list[str]) -> bool:
    return all(judgment_for(progress, row["image_id"], c) is not None
               for c in editable_checks(row, checks))


def cell_is_complete(rows_in_cell: list[dict], progress: dict) -> bool:
    return all(_row_checks_done(r, progress, CELL_CHECKS) for r in rows_in_cell)


def remaining_cells(rows: list[dict], progress: dict) -> int:
    """Number of (arm, font, variant) cells with at least one applicable footer
    check not yet judged. A cell whose checks are all n/a (clean) counts as
    done."""
    by_cell: dict[str, list[dict]] = {}
    for r in rows:
        by_cell.setdefault(cell_of(r["image_id"]), []).append(r)
    return sum(1 for cell in by_cell.values() if not cell_is_complete(cell, progress))


def sampled_image_ids(rows: list[dict]) -> list[str]:
    """The 40 Section 10 (c) body-sampled image_ids, in canonical order (the
    same sample make_manual_review drew when it built the CSV)."""
    return [rows[i]["image_id"] for i in sorted(mr.sampled_indices())]


def remaining_body(rows: list[dict], progress: dict) -> int:
    by_id = {r["image_id"]: r for r in rows}
    n = 0
    for iid in sampled_image_ids(rows):
        row = by_id[iid]
        if row.get(BODY_CHECK, "") == "n/a":
            continue
        if judgment_for(progress, iid, BODY_CHECK) is None:
            n += 1
    return n


# --------------------------------------------------------------------------- #
# crop boxes from the committed draw bboxes
# --------------------------------------------------------------------------- #
def _union(boxes: list[list[int]]) -> list[int] | None:
    if not boxes:
        return None
    ls, ts, rs, bs = zip(*boxes)
    return [min(ls), min(ts), max(rs), max(bs)]


def crop_box(entry: dict, kind: str, margin: int = CROP_MARGIN) -> list[int] | None:
    """The crop rectangle for `kind` from this image's draw bboxes, padded by
    `margin` and clamped to the image bounds. Returns None when the region does
    not exist (e.g. a footer crop for a clean image).

      footer   -- union of the footer draws.
      boundary -- union of the Total draw and the footer draws (shows the
                  body-to-footer spacing).
      body     -- union of the body and Total draws (the receipt body).
    """
    draws = entry.get("draws", [])
    if kind == "footer":
        boxes = [d["bbox"] for d in draws if d["kind"] == "footer"]
    elif kind == "boundary":
        # Only meaningful when there is a footer: it shows the Total-to-footer
        # gap. No footer (clean) -> no boundary.
        has_footer = any(d["kind"] == "footer" for d in draws)
        boxes = [d["bbox"] for d in draws if d["kind"] in ("total", "footer")] if has_footer else []
    elif kind == "body":
        boxes = [d["bbox"] for d in draws if d["kind"] in ("body", "total")]
    else:
        raise ValueError(f"unknown crop kind: {kind}")
    u = _union(boxes)
    if u is None:
        return None
    w, h = entry["width"], entry["height"]
    return [max(0, u[0] - margin), max(0, u[1] - margin),
            min(w, u[2] + margin), min(h, u[3] + margin)]


def draw_texts(entry: dict, kinds: tuple[str, ...]) -> list[str]:
    return [d["text"] for d in entry.get("draws", []) if d["kind"] in kinds]


def render_crop_png(image_path: Path, box: list[int] | None) -> bytes:
    from PIL import Image
    img = Image.open(image_path)
    img.load()
    if box is not None:
        img = img.crop(tuple(box))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# --export : merge progress into the CSV
# --------------------------------------------------------------------------- #
class ExportError(Exception):
    pass


def _canonical_ids() -> list[str]:
    return [e["image_id"] for e in mr.canonical_entries()]


def build_exported_rows(rows: list[dict], progress: dict) -> list[dict]:
    """The CSV rows with every editable (non-n/a) check filled from progress and
    the row's fail notes gathered into `notes`. Raises ExportError on any
    structural change, any still-empty required cell, or any bad value."""
    canonical = _canonical_ids()
    if len(rows) != len(canonical):
        raise ExportError(f"row count changed: {len(rows)} != {len(canonical)}")
    for i, (row, cid) in enumerate(zip(rows, canonical)):
        if row["image_id"] != cid:
            raise ExportError(f"row {i} id/order changed: {row['image_id']!r} != {cid!r}")

    out: list[dict] = []
    for row in rows:
        new = dict(row)
        iid = row["image_id"]
        fail_notes: list[str] = []
        for col in ALL_CHECKS:
            if row.get(col, "") == "n/a":
                new[col] = "n/a"
                continue
            j = judgment_for(progress, iid, col)
            if j is None:
                raise ExportError(f"empty required cell: {iid} / {col}")
            value, note = j["value"], (j.get("note") or "").strip()
            if value not in ("pass", "fail"):
                raise ExportError(f"bad value {value!r} at {iid} / {col}")
            if value == "fail" and not note:
                raise ExportError(f"fail without note: {iid} / {col}")
            new[col] = value
            if value == "fail":
                fail_notes.append(f"{col}: {note}")
        new["notes"] = "; ".join(fail_notes)
        # Final guard: every check cell is exactly pass / fail / n/a.
        for col in ALL_CHECKS:
            if new[col] not in VALID_VALUES:
                raise ExportError(f"non-{VALID_VALUES} value {new[col]!r} at {iid} / {col}")
        out.append(new)
    return out


def value_counts(rows: list[dict]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for col in ALL_CHECKS:
        c = {v: 0 for v in VALID_VALUES}
        c["(empty)"] = 0
        for r in rows:
            v = r.get(col, "")
            c[v if v in c else "(empty)"] = c.get(v if v in c else "(empty)", 0) + 1
        counts[col] = c
    return counts


def collect_fails(rows: list[dict]) -> list[tuple[str, str, str]]:
    """(image_id, column, note) for every fail, in row then column order."""
    fails = []
    for r in rows:
        for col in ALL_CHECKS:
            if r.get(col) == "fail":
                note = ""
                for piece in (r.get("notes") or "").split("; "):
                    if piece.startswith(f"{col}: "):
                        note = piece[len(col) + 2:]
                fails.append((r["image_id"], col, note))
    return fails


def write_rows(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


def export(csv_path: Path = CSV_PATH, progress_path: Path = PROGRESS_PATH,
           out_path: Path | None = None) -> dict:
    rows = load_rows(csv_path)
    progress = load_progress(progress_path)
    exported = build_exported_rows(rows, progress)
    write_rows(exported, out_path or csv_path)
    return {
        "rows": exported,
        "value_counts": value_counts(exported),
        "fails": collect_fails(exported),
    }


# --------------------------------------------------------------------------- #
# HTTP server
# --------------------------------------------------------------------------- #
PAGE_CSS = """
body{font:14px/1.5 system-ui,sans-serif;margin:0;background:#f5f5f5;color:#111}
header{background:#1f2933;color:#fff;padding:10px 16px;position:sticky;top:0;z-index:5}
header a{color:#8fd0ff;margin-right:14px;text-decoration:none}
main{padding:16px;max-width:1100px}
.row{background:#fff;border:1px solid #d0d0d0;border-radius:6px;margin:12px 0;padding:12px;display:flex;gap:16px}
.row .crops{flex:0 0 auto}
.row img.crop{display:block;border:1px solid #bbb;background:#fff;margin-bottom:6px;max-width:520px;cursor:pointer}
.row .panel{flex:1 1 auto;min-width:260px}
.iid{font-weight:600;font-family:ui-monospace,monospace}
.tr{background:#f0f4f8;border:1px solid #d6e0ea;border-radius:4px;padding:6px 8px;margin:6px 0;white-space:pre-wrap;font-family:ui-monospace,monospace;font-size:12px}
.check{margin:6px 0;display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.check .name{flex:0 0 220px;font-family:ui-monospace,monospace}
button.j{border:1px solid #888;background:#fff;border-radius:4px;padding:3px 10px;cursor:pointer}
button.j.pass.on{background:#1a7f37;color:#fff;border-color:#1a7f37}
button.j.fail.on{background:#b42318;color:#fff;border-color:#b42318}
.locked{color:#888;font-style:italic}
.note{flex:1 1 240px;min-width:200px}
.cellline{margin:4px 0}
.cellline a{text-decoration:none}
.done{color:#1a7f37}.todo{color:#b42318}
.counter{font-weight:600}
"""

PAGE_JS = """
async function judge(iid, col, value, btn){
  let note="";
  const noteEl=document.getElementById('note|'+iid+'|'+col);
  if(value==='fail'){
    note=(noteEl&&noteEl.value||'').trim();
    if(!note){ alert('A fail requires a note.'); if(noteEl) noteEl.focus(); return; }
  } else if(noteEl){ note=noteEl.value.trim(); }
  const r=await fetch('/judge',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({image_id:iid,column:col,value:value,note:note})});
  const d=await r.json();
  if(!d.ok){ alert(d.error||'error'); return; }
  const grp=document.querySelectorAll('[data-k="'+iid+'|'+col+'"]');
  grp.forEach(b=>b.classList.remove('on'));
  btn.classList.add('on');
  const rc=document.getElementById('remaining'); if(rc) rc.textContent=d.remaining_cells;
  const rb=document.getElementById('remaining_body'); if(rb) rb.textContent=d.remaining_body;
}
"""


def _html_head(title: str) -> str:
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title>"
            f"<style>{PAGE_CSS}</style><script>{PAGE_JS}</script></head><body>")


def _esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def _check_controls(row: dict, progress: dict, checks: list[str]) -> str:
    iid = row["image_id"]
    out = []
    for col in checks:
        if row.get(col, "") == "n/a":
            out.append(f"<div class='check'><span class='name'>{col}</span>"
                       f"<span class='locked'>n/a (locked)</span></div>")
            continue
        j = judgment_for(progress, iid, col)
        val = j["value"] if j else ""
        note = _esc(j["note"]) if j else ""
        k = f"{iid}|{col}"
        pon = " on" if val == "pass" else ""
        fon = " on" if val == "fail" else ""
        out.append(
            f"<div class='check'><span class='name'>{col}</span>"
            f"<button class='j pass{pon}' data-k='{k}' onclick=\"judge('{iid}','{col}','pass',this)\">pass</button>"
            f"<button class='j fail{fon}' data-k='{k}' onclick=\"judge('{iid}','{col}','fail',this)\">fail</button>"
            f"<input class='note' id='note|{k}' placeholder='note (required on fail)' value='{note}'>"
            f"</div>")
    return "".join(out)


def _cell_rows_html(cell: str, rows: list[dict], g3a: dict, progress: dict) -> str:
    out = []
    for row in rows:
        iid = row["image_id"]
        entry = g3a[iid]
        footer_lines = "\n".join(draw_texts(entry, ("footer",))) or "(no footer)"
        fcrop = f"/crop?id={iid}&kind=footer"
        bcrop = f"/crop?id={iid}&kind=boundary"
        out.append(
            f"<div class='row'><div class='crops'>"
            f"<img class='crop' src='{fcrop}' title='footer' onclick=\"window.open('/full?id={iid}')\">"
            f"<img class='crop' src='{bcrop}' title='TOTAL-to-footer boundary' onclick=\"window.open('/full?id={iid}')\">"
            f"</div><div class='panel'>"
            f"<div class='iid'>{iid}</div>"
            f"<div class='tr'>{_esc(footer_lines)}</div>"
            f"{_check_controls(row, progress, CELL_CHECKS)}"
            f"</div></div>")
    return "".join(out)


def _body_rows_html(rows_by_id: dict, sampled: list[str], g3a: dict, progress: dict) -> str:
    out = []
    for iid in sampled:
        row = rows_by_id[iid]
        entry = g3a[iid]
        body_lines = "\n".join(draw_texts(entry, ("body", "total"))) or "(no body)"
        bcrop = f"/crop?id={iid}&kind=body"
        out.append(
            f"<div class='row'><div class='crops'>"
            f"<img class='crop' src='{bcrop}' title='body' onclick=\"window.open('/full?id={iid}')\">"
            f"</div><div class='panel'>"
            f"<div class='iid'>{iid}</div>"
            f"<div class='tr'>{_esc(body_lines)}</div>"
            f"{_check_controls(row, progress, [BODY_CHECK])}"
            f"</div></div>")
    return "".join(out)


def _index_html(rows: list[dict], g3a: dict, progress: dict) -> str:
    by_cell: dict[str, list[dict]] = {}
    for r in rows:
        by_cell.setdefault(cell_of(r["image_id"]), []).append(r)
    lines = []
    for cell in cell_order(rows):
        done = cell_is_complete(by_cell[cell], progress)
        n_edit = len(editable_checks(by_cell[cell][0], CELL_CHECKS))
        tag = "<span class='done'>done</span>" if done else "<span class='todo'>todo</span>"
        extra = " (all n/a)" if n_edit == 0 else f" ({n_edit} checks x 40)"
        lines.append(f"<div class='cellline'>{tag} <a href='/cell?c={cell}'>{cell}</a>{extra}</div>")
    body = (f"<h2>Cells</h2>"
            f"<p>Cells remaining: <span class='counter' id='remaining'>{remaining_cells(rows, progress)}</span>"
            f" / {len(by_cell)}</p>"
            + "".join(lines)
            + f"<h2>Body (Section 10 (c) sample)</h2>"
            f"<p>Body images remaining: <span class='counter' id='remaining_body'>{remaining_body(rows, progress)}</span>"
            f" / {len(sampled_image_ids(rows))} &nbsp; <a href='/body'>open body view</a></p>"
            f"<p>When every cell and body image is done, run "
            f"<code>python3 phase3b/g3_review.py --export</code>.</p>")
    return _html_head("Phase 3b G3 review") + _header() + "<main>" + body + "</main></body></html>"


def _header() -> str:
    return ("<header><a href='/'>index</a><a href='/body'>body</a>"
            "<span>Phase 3b G3 (b)+(c) review &mdash; localhost only, no bulk-pass</span></header>")


class ReviewHandler(BaseHTTPRequestHandler):
    rows: list[dict] = []
    g3a: dict = {}
    progress: dict = {}
    progress_path: Path = PROGRESS_PATH

    def log_message(self, *a):  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, html: str, code: int = 200):
        self._send(code, html.encode("utf-8"), "text/html; charset=utf-8")

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        cls = type(self)
        rows_by_id = {r["image_id"]: r for r in cls.rows}
        if u.path == "/":
            self._html(_index_html(cls.rows, cls.g3a, cls.progress))
        elif u.path == "/cell":
            cell = q.get("c", [""])[0]
            rows = [r for r in cls.rows if cell_of(r["image_id"]) == cell]
            if not rows:
                self._html("<p>unknown cell</p>", 404); return
            html = (_html_head(f"cell {cell}") + _header() + "<main>"
                    + f"<h2>{_esc(cell)}</h2>"
                    + _cell_rows_html(cell, rows, cls.g3a, cls.progress)
                    + "</main></body></html>")
            self._html(html)
        elif u.path == "/body":
            sampled = sampled_image_ids(cls.rows)
            html = (_html_head("body view") + _header() + "<main>"
                    + "<h2>Body line-by-line (40 sampled)</h2>"
                    + _body_rows_html(rows_by_id, sampled, cls.g3a, cls.progress)
                    + "</main></body></html>")
            self._html(html)
        elif u.path == "/crop":
            iid = q.get("id", [""])[0]
            kind = q.get("kind", ["footer"])[0]
            entry = cls.g3a.get(iid)
            if entry is None:
                self._send(404, b"no such image", "text/plain"); return
            box = crop_box(entry, kind)
            self._send(200, render_crop_png(_image_path_for(iid), box), "image/png")
        elif u.path == "/full":
            iid = q.get("id", [""])[0]
            if iid not in cls.g3a:
                self._send(404, b"no such image", "text/plain"); return
            self._send(200, render_crop_png(_image_path_for(iid), None), "image/png")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/judge":
            self._send(404, b"not found", "text/plain"); return
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        cls = type(self)
        try:
            record_judgment(cls.progress, payload["image_id"], payload["column"],
                            payload["value"], payload.get("note", ""))
            save_progress(cls.progress, cls.progress_path)
            resp = {"ok": True,
                    "remaining_cells": remaining_cells(cls.rows, cls.progress),
                    "remaining_body": remaining_body(cls.rows, cls.progress)}
        except (ValueError, KeyError) as e:
            resp = {"ok": False, "error": str(e)}
        self._send(200, json.dumps(resp).encode("utf-8"), "application/json")


def _image_path_for(image_id: str) -> Path:
    """Committed PNG path for an image_id (A|v|rid -> rid_v.png;
    B|font|v|rid -> rid_v_font.png)."""
    p = parse_image_id(image_id)
    if p["arm"] == "A":
        name = f"{p['receipt_id']}_{p['variant']}.png"
    else:
        name = f"{p['receipt_id']}_{p['variant']}_{p['font']}.png"
    return IMAGES_DIR / name


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    ReviewHandler.rows = load_rows()
    ReviewHandler.g3a = load_g3a()
    ReviewHandler.progress = load_progress()
    ReviewHandler.progress_path = PROGRESS_PATH
    httpd = ThreadingHTTPServer((host, port), ReviewHandler)
    print(f"Phase 3b G3 review: http://{host}:{port}/  (Ctrl-C to stop)")
    print(f"progress -> {PROGRESS_PATH}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


def _print_export(summary: dict) -> None:
    print("Value counts per column:")
    for col, counts in summary["value_counts"].items():
        shown = ", ".join(f"{v}={counts[v]}" for v in ("pass", "fail", "n/a", "(empty)")
                          if counts.get(v))
        print(f"  {col:<32} {shown}")
    print(f"\nFails ({len(summary['fails'])}):")
    if not summary["fails"]:
        print("  (none)")
    for iid, col, note in summary["fails"]:
        print(f"  FAIL  {iid}  {col}: {note}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", action="store_true",
                        help="merge progress into the CSV and print the summary")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--progress", type=Path, default=PROGRESS_PATH)
    parser.add_argument("--out", type=Path, default=None,
                        help="export destination (default: overwrite the CSV)")
    args = parser.parse_args(argv)
    if args.export:
        try:
            summary = export(args.csv, args.progress, args.out)
        except ExportError as e:
            print(f"REFUSED: {e}", file=sys.stderr)
            return 2
        _print_export(summary)
        print(f"\nWrote {len(summary['rows'])} rows to {args.out or args.csv}")
        return 0
    serve(args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
