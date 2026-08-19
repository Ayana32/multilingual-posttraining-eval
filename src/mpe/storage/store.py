from __future__ import annotations

from pathlib import Path

from mpe.storage.schema import ResultRecord


class ResultStore:
    """Append-only JSONL storage for ResultRecords, one file per run_id.

    JSONL (not a single growing table) so a run can be written incrementally
    and so results/ stays trivially diffable/greppable -- deliberately the
    simplest thing that supports Phase 1's needs; a parquet/DB-backed store
    is a Phase 4+ concern if/when result volume actually demands it.
    """

    def __init__(self, results_dir: str | Path = "results"):
        self.results_dir = Path(results_dir)

    def path_for_run(self, run_id: str) -> Path:
        return self.results_dir / run_id / "records.jsonl"

    def write(self, records: list[ResultRecord]) -> Path:
        if not records:
            raise ValueError("write() called with an empty records list")
        run_ids = {r.run_id for r in records}
        if len(run_ids) > 1:
            raise ValueError(f"write() received records from multiple runs: {run_ids}")
        path = self.path_for_run(records[0].run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for record in records:
                f.write(record.model_dump_json() + "\n")
        return path

    def read(self, run_id: str) -> list[ResultRecord]:
        path = self.path_for_run(run_id)
        if not path.exists():
            return []
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(ResultRecord.model_validate_json(line))
        return records

    def list_runs(self) -> list[str]:
        if not self.results_dir.exists():
            return []
        return sorted(p.parent.name for p in self.results_dir.glob("*/records.jsonl"))
