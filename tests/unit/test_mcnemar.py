from datetime import datetime, timezone

import pytest

from mpe.analysis.bootstrap import BootstrapComparabilityError
from mpe.analysis.mcnemar import exact_mcnemar_p, mcnemar_paired
from mpe.checkpoints.schema import CheckpointStage
from mpe.datasets.schema import TaskType
from mpe.evaluators.schema import GenerationConfig
from mpe.storage.schema import ResultRecord
from mpe.storage.store import ResultStore


def _record(
    run_id: str, item_id: str, parallel_item_id: str, refusal: bool | None, label: str = "harmful", **overrides
) -> ResultRecord:
    fields = dict(
        run_id=run_id,
        experiment_name="exp",
        lineage="instruct",
        stage=CheckpointStage.SFT,
        hf_repo_id="x",
        revision="main",
        language="en",
        benchmark="polyguard_prompts",
        item_id=item_id,
        parallel_item_id=parallel_item_id,
        task_type=TaskType.REFUSAL_CLASSIFICATION,
        prompt="p",
        completion="c",
        is_parseable=True,
        language_match=True,
        mc_correct=None,
        refusal_label=None,
        behavior_matches_expected=None,
        expected_label=label,
        scorer_response_refusal=refusal,
        scorer_parse_ok=True if refusal is not None else None,
        generation_config=GenerationConfig(),
        created_at=datetime.now(timezone.utc),
    )
    fields.update(overrides)
    return ResultRecord(**fields)


class TestExactMcNemarP:
    def test_no_discordant_pairs_is_p_one(self):
        assert exact_mcnemar_p(0, 0) == 1.0

    def test_balanced_discordance_is_p_one(self):
        assert exact_mcnemar_p(3, 3) == 1.0

    def test_known_value_all_one_direction(self):
        # n=10, k=0: 2 * (1/1024)
        assert exact_mcnemar_p(0, 10) == pytest.approx(2 / 1024)

    def test_known_value_one_vs_nine(self):
        # n=10, k=1: 2 * (1 + 10) / 1024
        assert exact_mcnemar_p(1, 9) == pytest.approx(22 / 1024)

    def test_symmetric(self):
        assert exact_mcnemar_p(2, 7) == exact_mcnemar_p(7, 2)

    def test_negative_counts_rejected(self):
        with pytest.raises(ValueError):
            exact_mcnemar_p(-1, 3)


class TestMcNemarPaired:
    def _write_pair(self, store, en_pattern, ko_pattern):
        store.write([_record("en", f"e{i}", str(i), v, language="en") for i, v in enumerate(en_pattern)])
        store.write([_record("ko", f"k{i}", str(i), v, language="ko") for i, v in enumerate(ko_pattern)])

    def test_counts_cells_by_parallel_id(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        en = [True, True, True, False, True]
        ko = [True, False, False, False, True]
        self._write_pair(store, en, ko)
        r = mcnemar_paired(store, "en", "ko")
        assert (r.both_refused, r.only_a_refused, r.only_b_refused, r.neither_refused) == (2, 2, 0, 1)
        assert r.n_paired_items == 5
        assert r.p_value_exact == pytest.approx(exact_mcnemar_p(2, 0))

    def test_none_verdicts_are_excluded_and_reported(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        self._write_pair(store, [True, None, True], [False, False, None])
        r = mcnemar_paired(store, "en", "ko")
        assert r.n_excluded_pairs == 2
        assert r.n_paired_items == 1

    def test_unharmful_items_are_ignored_for_harmful_condition(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        store.write([
            _record("en", "e0", "0", True, language="en"),
            _record("en", "e1", "1", True, label="unharmful", language="en"),
        ])
        store.write([
            _record("ko", "k0", "0", False, language="ko"),
            _record("ko", "k1", "1", False, label="unharmful", language="ko"),
        ])
        r = mcnemar_paired(store, "en", "ko", condition="harmful")
        assert r.n_paired_items == 1 and r.only_a_refused == 1

    def test_mixed_language_and_stage_difference_is_rejected(self, tmp_path):
        store = ResultStore(results_dir=tmp_path)
        store.write([_record("en_sft", "e0", "0", True, language="en")])
        store.write([_record("ko_dpo", "k0", "0", False, language="ko", stage=CheckpointStage.DPO)])
        with pytest.raises(BootstrapComparabilityError):
            mcnemar_paired(store, "en_sft", "ko_dpo")
