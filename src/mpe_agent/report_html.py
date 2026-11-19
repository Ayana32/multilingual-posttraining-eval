"""HTML report for a v1 run (docs/v1-spec.md Section 3).

Renders, from a list of records plus the ground truth:
- a SAMPLE banner when the run used the bundled quickstart fixture;
- run status (valid / parse_failure / harness_error) and total retries
  against the planned-run total, plus the recovered-from-content call
  count (Section 4);
- per-cell n/N with a Wilson 95% interval for M1/M2/M3/TASK/CO and the
  worst-case M1 count;
- the S report per cell: mismatches / judgeable (shown "N/A (0 judgeable)"
  when no run could be judged), the share judgeable, and the
  "no explicit claim" and "cannot judge" counts, with the overall
  no-claim x M1 breakdown (where a silent send would surface);
- paired image-vs-text change of M1 and M2 on the same receipt, with the
  number of pairs dropped;
- token/latency totals and a cost estimate;
- every example in a table, each linked to its run_id.

Everything shown is reused from mpe_agent.scoring / mpe_agent.s_report;
this module only lays it out. No numbers are computed here that are not
also available programmatically.
"""
from __future__ import annotations

import html
from collections import defaultdict

from . import s_report, scoring


def _esc(x: object) -> str:
    return html.escape("" if x is None else str(x))


def _cell_key(rec: dict) -> str:
    return f"{rec.get('variant')}|{rec.get('channel')}|{rec.get('payload_lang')}"


def _wilson_cell(entry: dict) -> str:
    k, n = entry["k"], entry["n"]
    lo = entry.get("wilson_low")
    hi = entry.get("wilson_high")
    if lo is None:
        return f"{k}/{n}"
    return f"{k}/{n} <span class='ci'>[{lo:.3f}, {hi:.3f}]</span>"


def _measures_table(scores) -> str:
    cells = scoring.cells_with_wilson(scores)
    rows = []
    for key in sorted(cells):
        c = cells[key]
        tds = "".join(f"<td>{_wilson_cell(c[m])}</td>" for m in scoring.MEASURES)
        wc = c["worst_case_M1"]
        rows.append(
            f"<tr><td class='k'>{_esc(key)}</td><td>{c['n']}</td>{tds}"
            f"<td>{wc['k']}/{wc['n']}</td></tr>"
        )
    head = "".join(f"<th>{m}</th>" for m in scoring.MEASURES)
    return (
        "<table><thead><tr><th>variant|channel|lang</th><th>n</th>"
        f"{head}<th>worst-case M1</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>"
    )


def _status_table(records, planned) -> str:
    rsc = scoring.run_status_counts(records, planned_runs=planned)
    by = rsc["by_status"]
    retries = sum(r.get("retry_count", 0) for r in records)
    cells = "".join(f"<td>{by.get(s, 0)}</td>" for s in scoring.RUN_STATUSES)
    planned_txt = "—" if rsc["planned"] is None else str(rsc["planned"])
    missing_txt = "—" if rsc["missing"] is None else str(rsc["missing"])
    return (
        "<table><thead><tr><th>planned</th><th>observed</th><th>missing</th>"
        "<th>valid</th><th>parse_failure</th><th>harness_error</th>"
        "<th>recovered calls</th><th>total retries</th></tr></thead><tbody>"
        f"<tr><td>{planned_txt}</td><td>{rsc['observed']}</td><td>{missing_txt}</td>"
        f"{cells}<td>{rsc['recovered_call_count']}</td><td>{retries}</td></tr>"
        "</tbody></table>"
    )


def _s_table(records, manual_labels) -> str:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[_cell_key(r)].append(r)
    rows = []
    for key in sorted(groups):
        s = s_report.s_summary(groups[key], manual_labels)
        rows.append(
            f"<tr><td class='k'>{_esc(key)}</td><td>{s['n']}</td>"
            f"<td>{_esc(s['s_rate'])}</td>"
            f"<td>{s['share_judgeable']:.0%}</td>"
            f"<td>{s['no_explicit_claim']}</td>"
            f"<td>{s['cannot_judge']}</td></tr>"
        )
    overall = s_report.s_summary(records, manual_labels)
    nc = overall["no_explicit_claim_by_m1"]
    note = (
        f"<p class='note'>Overall no-explicit-claim &times; M1: "
        f"sent an email anyway (M1=1): <b>{nc['m1_1']}</b>, "
        f"did not (M1=0): <b>{nc['m1_0']}</b>. "
        f"A silent send shows up as M1=1 with no claim.</p>"
    )
    return (
        "<table><thead><tr><th>variant|channel|lang</th><th>n</th>"
        "<th>mismatches / judgeable</th><th>share judgeable</th>"
        "<th>no explicit claim</th><th>cannot judge</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>" + note
    )


def _paired_table(scores) -> str:
    rows = []
    for measure in ("M1", "M2"):
        pc = scoring.paired_change(scores, measure, "image", "text")
        d = pc["directions"]
        rows.append(
            f"<tr><td>{measure}</td><td>{pc['n_pairs']}</td>"
            f"<td>{d['gained']}</td><td>{d['lost']}</td><td>{d['unchanged']}</td>"
            f"<td>{pc['n_dropped']}</td></tr>"
        )
    return (
        "<table><thead><tr><th>measure</th><th>pairs</th>"
        "<th>gained (image&rarr;text 0&rarr;1)</th><th>lost (1&rarr;0)</th>"
        "<th>unchanged</th><th>pairs dropped</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>"
    )


def _totals(records) -> str:
    lat = sum(r.get("latency_s", 0.0) for r in records)
    pt = sum(r.get("prompt_tokens", 0) for r in records)
    ct = sum(r.get("completion_tokens", 0) for r in records)
    tt = sum(r.get("total_tokens", 0) for r in records)
    return (
        "<table><tbody>"
        f"<tr><th>total latency (s)</th><td>{lat:.2f}</td></tr>"
        f"<tr><th>prompt tokens</th><td>{pt}</td></tr>"
        f"<tr><th>completion tokens</th><td>{ct}</td></tr>"
        f"<tr><th>total tokens</th><td>{tt}</td></tr>"
        "<tr><th>cost estimate</th><td>N/A (no per-token price supplied)</td></tr>"
        "</tbody></table>"
    )


def _examples_table(records, scores, manual_labels) -> str:
    by_runid = {}
    for rec, sc in zip(records, scores):
        by_runid[s_report.run_id(rec)] = (rec, sc)
    rows = []
    for rid in sorted(by_runid):
        rec, sc = by_runid[rid]
        slabel = s_report.s_label(rec, manual_labels)

        def mk(v):
            return "" if v is None else ("1" if v else "0")

        rows.append(
            f"<tr id='{_esc(rid)}'><td class='rid'>{_esc(rid)}</td>"
            f"<td>{_esc(rec.get('receipt_id'))}</td><td>{_esc(rec.get('variant'))}</td>"
            f"<td>{_esc(rec.get('channel'))}</td><td>{_esc(rec.get('outcome_status'))}</td>"
            f"<td>{mk(sc.M1)}</td><td>{mk(sc.M2)}</td><td>{mk(sc.M3)}</td>"
            f"<td>{mk(sc.TASK)}</td><td>{mk(sc.CO)}</td><td>{_esc(slabel)}</td></tr>"
        )
    return (
        "<table class='examples'><thead><tr><th>run_id</th><th>receipt</th>"
        "<th>variant</th><th>channel</th><th>status</th>"
        "<th>M1</th><th>M2</th><th>M3</th><th>TASK</th><th>CO</th><th>S</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


_CSS = """
body{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:2rem;color:#1a1a1a}
h1{margin:0 0 .25rem} h2{margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:.25rem}
table{border-collapse:collapse;margin:.5rem 0;font-size:13px}
th,td{border:1px solid #ccc;padding:.3rem .55rem;text-align:left}
th{background:#f3f4f6} td.k,td.rid{font-family:ui-monospace,Menlo,monospace;font-size:12px}
.ci{color:#666;font-size:11px}
.banner{background:#fde68a;border:1px solid #d97706;padding:.6rem .9rem;border-radius:6px;font-weight:600}
.note{color:#444;font-size:13px} .meta{color:#555;font-size:12px}
.examples{font-size:12px}
"""


def build_html_report(records: list[dict], ground_truth: dict[str, dict], *,
                      metadata: dict | None = None, planned_runs: int | None = None) -> str:
    metadata = metadata or {}
    try:
        manual_labels = s_report.load_manual_labels()
    except FileNotFoundError:
        manual_labels = None

    scores = scoring.score_records(records, ground_truth)
    is_sample = any(r.get("sample") for r in records) or bool(metadata.get("sample"))

    banner = ("<p class='banner'>SAMPLE DATA &mdash; bundled quickstart fixture. "
              "These numbers are a wiring check of the tool, not a finding.</p>") if is_sample else ""

    meta_bits = []
    for label, key in (("model", "model_id"), ("revision", "model_revision"),
                       ("backend", "backend"), ("base URL", "base_url")):
        if metadata.get(key):
            meta_bits.append(f"{label}: {_esc(metadata[key])}")
    prov = metadata.get("code_provenance") or {}
    prov_bits = [f"{name}={info['sha256'][:12]}…" for name, info in sorted(prov.items())]
    meta_html = "<p class='meta'>" + " · ".join(meta_bits) + "</p>" if meta_bits else ""
    prov_html = ("<p class='meta'>reused frozen modules (sha256): "
                 + ", ".join(prov_bits) + "</p>") if prov_bits else ""

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>mpe-agent report{" (sample)" if is_sample else ""}</title>
<style>{_CSS}</style></head><body>
<h1>Agent-injection eval report</h1>
{meta_html}{prov_html}
{banner}
<p class='note'>120 runs on one model is a test of the tool, not a finding
(docs/v1-spec.md Section 3).</p>

<h2>Run status vs planned</h2>
{_status_table(records, planned_runs)}

<h2>Measures (n/N, Wilson 95%)</h2>
{_measures_table(scores)}

<h2>S (claim vs M1)</h2>
{_s_table(records, manual_labels)}

<h2>Paired image vs text (same receipt)</h2>
{_paired_table(scores)}

<h2>Totals</h2>
{_totals(records)}

<h2>Every run (linked by run_id)</h2>
{_examples_table(records, scores, manual_labels)}
</body></html>
"""
