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
from mpe.scorers.base import Scorer
from mpe.storage.schema import ResultRecord
from mpe.storage.store import ResultStore


class ExperimentRunner:
    """Orchestrates one ExperimentConfig into stored ResultRecords.

    Sequencing: registry lookup -> dataset load -> generate -> parse -> score
    -> (optional authoritative scorer) -> store, for every (stage, benchmark,
    language) combination the config asks for. Nothing here re-implements
    logic that belongs in a lower module -- this class's job is sequencing
    and building ResultRecord, not generation, parsing, or scoring.

    Writes to ResultStore after each (stage, benchmark, language) batch
    completes, not once at the very end. With a mock evaluator this
    distinction is invisible (nothing fails), but with a real model a single
    bad item several hours into a run must not cost every already-computed
    result -- see HFTransformersEvaluator's per-item try/except for the other
    half of this failure-isolation story.
    """

    def __init__(
        self,
        checkpoint_registry: CheckpointRegistry,
        evaluator: Evaluator,
        result_store: ResultStore,
        loaders: dict[str, DatasetLoader] | None = None,
        scorer: Scorer | None = None,
    ):
        self.checkpoint_registry = checkpoint_registry
        self.evaluator = evaluator
        self.result_store = result_store
        self.loaders = loaders or {name: cls() for name, cls in LOADERS.items()}
        self.scorer = scorer

    def run(self, config: ExperimentConfig) -> str:
        run_id = f"{config.name}_{uuid.uuid4().hex[:8]}"
        lineage = self.checkpoint_registry.get_lineage(config.lineage)

        for benchmark_name in config.benchmarks:
            if benchmark_name not in self.loaders:
                raise KeyError(
                    f"Unknown benchmark '{benchmark_name}'. "
                    f"Known benchmarks: {list(self.loaders)}"
                )

        any_records = False
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

                    batch: list[ResultRecord] = []
                    for item in items:
                        response = responses_by_id[item.item_id]
                        parsed = parse_output(item, response)
                        scored = score_item(item, parsed)

                        verdict = self.scorer.score(item, response) if self.scorer else None

                        batch.append(
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
                                finish_reason=response.finish_reason,
                                generation_protocol=response.generation_protocol,
                                is_parseable=scored.is_parseable,
                                language_match=scored.language_match,
                                mc_correct=scored.mc_correct,
                                refusal_label=scored.refusal_label,
                                behavior_matches_expected=scored.behavior_matches_expected,
                                expected_label=item.expected_label,
                                scorer_name=verdict.scorer_name if verdict else None,
                                scorer_prompt_harmful=verdict.prompt_harmful if verdict else None,
                                scorer_response_refusal=verdict.response_refusal if verdict else None,
                                scorer_response_harmful=verdict.response_harmful if verdict else None,
                                scorer_parse_ok=verdict.parse_ok if verdict else None,
                                scorer_raw_output=verdict.raw_output if verdict else None,
                                generation_config=config.generation,
                                created_at=datetime.now(timezone.utc),
                            )
                        )

                    self.result_store.write(batch)
                    any_records = True

        if not any_records:
            raise ValueError(
                "Experiment produced zero records -- every (stage, benchmark, "
                "language) combination returned no items. Nothing was written."
            )
        return run_id
