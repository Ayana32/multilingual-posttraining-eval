from __future__ import annotations

import uuid
from datetime import datetime, timezone

from mpe.checkpoints.registry import CheckpointRegistry
from mpe.config.experiment import ExperimentConfig
from mpe.datasets import LOADERS
from mpe.datasets.base import DatasetLoader
from mpe.evaluators.base import Evaluator
from mpe.metrics.core import score_item
from mpe.parsing.output_parser import parse_output
from mpe.storage.schema import ResultRecord
from mpe.storage.store import ResultStore


class ExperimentRunner:
    """Orchestrates one ExperimentConfig into stored ResultRecords.

    The only orchestration loop in Phase 1: registry lookup -> dataset load
    -> generate -> parse -> score -> store, for every (stage, benchmark,
    language) combination the config asks for. Nothing here re-implements
    logic that belongs in a lower module -- this class's job is sequencing
    and building ResultRecord, not generation, parsing, or scoring.
    """

    def __init__(
        self,
        checkpoint_registry: CheckpointRegistry,
        evaluator: Evaluator,
        result_store: ResultStore,
        loaders: dict[str, DatasetLoader] | None = None,
    ):
        self.checkpoint_registry = checkpoint_registry
        self.evaluator = evaluator
        self.result_store = result_store
        self.loaders = loaders or {name: cls() for name, cls in LOADERS.items()}

    def run(self, config: ExperimentConfig) -> str:
        run_id = f"{config.name}_{uuid.uuid4().hex[:8]}"
        lineage = self.checkpoint_registry.get_lineage(config.lineage)

        for benchmark_name in config.benchmarks:
            if benchmark_name not in self.loaders:
                raise KeyError(
                    f"Unknown benchmark '{benchmark_name}'. "
                    f"Known benchmarks: {list(self.loaders)}"
                )

        records: list[ResultRecord] = []
        for stage in config.stages:
            checkpoint = lineage.get_stage(stage)
            for benchmark_name in config.benchmarks:
                loader = self.loaders[benchmark_name]
                for language in config.languages:
                    items = loader.load(
                        language, limit=config.limit_per_benchmark, seed=config.seed
                    )
                    if not items:
                        continue

                    responses = self.evaluator.generate(checkpoint, items, config.generation)
                    responses_by_id = {r.item_id: r for r in responses}

                    for item in items:
                        response = responses_by_id[item.item_id]
                        parsed = parse_output(item, response)
                        scored = score_item(item, parsed)

                        records.append(
                            ResultRecord(
                                run_id=run_id,
                                experiment_name=config.name,
                                lineage=config.lineage,
                                stage=stage,
                                hf_repo_id=checkpoint.hf_repo_id,
                                revision=checkpoint.revision,
                                language=language,
                                benchmark=benchmark_name,
                                item_id=item.item_id,
                                parallel_item_id=item.parallel_item_id,
                                task_type=item.task_type,
                                prompt=item.prompt,
                                completion=response.completion,
                                is_parseable=scored.is_parseable,
                                language_match=scored.language_match,
                                mc_correct=scored.mc_correct,
                                refusal_label=scored.refusal_label,
                                behavior_matches_expected=scored.behavior_matches_expected,
                                expected_label=item.expected_label,
                                generation_config=config.generation,
                                created_at=datetime.now(timezone.utc),
                            )
                        )

        self.result_store.write(records)
        return run_id
