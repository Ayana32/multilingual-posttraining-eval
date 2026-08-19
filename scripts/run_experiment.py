#!/usr/bin/env python3
"""CLI entry point: run one experiment config end-to-end.

Example (mock evaluator, no network/GPU needed after first dataset cache):
    python scripts/run_experiment.py \
        --experiment-config configs/experiments/smoke_test.yaml \
        --evaluator mock
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mpe.checkpoints.registry import CheckpointRegistry  # noqa: E402
from mpe.config.experiment import ExperimentConfig  # noqa: E402
from mpe.evaluators.base import Evaluator  # noqa: E402
from mpe.evaluators.mock import MockEvaluator  # noqa: E402
from mpe.runner.experiment_runner import ExperimentRunner  # noqa: E402
from mpe.storage.store import ResultStore  # noqa: E402


def build_evaluator(name: str) -> Evaluator:
    if name == "mock":
        return MockEvaluator()
    if name == "hf":
        from mpe.evaluators.hf_transformers import HFTransformersEvaluator

        return HFTransformersEvaluator()
    raise ValueError(f"Unknown evaluator '{name}'")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a multilingual post-training safety evaluation experiment."
    )
    parser.add_argument("--experiment-config", required=True, type=Path)
    parser.add_argument(
        "--checkpoint-registry",
        default=Path("configs/models/olmo3_lineages.yaml"),
        type=Path,
    )
    parser.add_argument("--results-dir", default=Path("results"), type=Path)
    parser.add_argument("--evaluator", choices=["mock", "hf"], default="mock")
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.experiment_config)
    registry = CheckpointRegistry.from_yaml(args.checkpoint_registry)
    evaluator = build_evaluator(args.evaluator)
    store = ResultStore(args.results_dir)
    runner = ExperimentRunner(registry, evaluator, store)

    run_id = runner.run(config)
    records = store.read(run_id)
    print(f"Run complete: {run_id}")
    print(f"Records written: {len(records)}")
    print(f"Results file: {store.path_for_run(run_id)}")


if __name__ == "__main__":
    main()
