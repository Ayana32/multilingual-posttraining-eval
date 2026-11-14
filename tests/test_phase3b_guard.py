"""Phase 3b eval guard (docs/phase3b-replication.md Section 4a), mirroring the
Phase 3 guard tests. Exercised against throwaway git repos with their own
'phase3b-prereg' tag, never this project's real repo (CI's checkout does not
fetch tags, and the real tag state should not decide a refusal-path test).

Covers:
- guard_phase3b_eval_run allows rows outside receipt_set == 'phase3b_eval'
  (tune, Phase 3's own 'eval') unconditionally -- it must not interfere with
  Phase 3's guard.
- it refuses a 'phase3b_eval' row when the tag is missing, when HEAD is before
  the tag, and on a diverged branch; it passes at the tag commit and at a
  descendant.
- start_run's dirty-tree refusal now also covers 'phase3b_eval', while Phase
  3's 'eval' behaviour is unchanged and 'tune' is still allowed dirty.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "phase3"))

import harness as hz  # noqa: E402


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _build_phase3b_repo(repo: Path) -> dict[str, str]:
    """A -> B (tagged phase3b-prereg) -> C, HEAD left at C (a descendant)."""
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")

    (repo / "f.txt").write_text("a\n", encoding="utf-8")
    _git(repo, "add", "."); _git(repo, "commit", "-q", "-m", "A")
    commit_a = _git(repo, "rev-parse", "HEAD")

    (repo / "f.txt").write_text("b\n", encoding="utf-8")
    _git(repo, "add", "."); _git(repo, "commit", "-q", "-m", "B")
    commit_b = _git(repo, "rev-parse", "HEAD")
    _git(repo, "tag", hz.PHASE3B_TAG)

    (repo / "f.txt").write_text("c\n", encoding="utf-8")
    _git(repo, "add", "."); _git(repo, "commit", "-q", "-m", "C")
    commit_c = _git(repo, "rev-parse", "HEAD")
    return {"A": commit_a, "B": commit_b, "C": commit_c}


# ---- guard allows rows outside phase3b_eval, unconditionally ----

@pytest.mark.parametrize("receipt_set", ["tune", "eval", "other"])
def test_guard_allows_non_phase3b_rows(tmp_path, receipt_set):
    # No tag, bare repo: a non-phase3b_eval row must still pass (early return),
    # so the Phase 3b guard never interferes with Phase 3's own rows.
    _git(tmp_path, "init", "-q")
    hz.guard_phase3b_eval_run({"receipt_set": receipt_set}, repo_root=tmp_path)


# ---- tag / descendant checks ----

def test_phase3b_tag_exists_and_head_checks(tmp_path):
    commits = _build_phase3b_repo(tmp_path)
    assert hz.phase3b_tag_exists(tmp_path) is True
    assert hz.head_is_phase3b_or_descendant(tmp_path) is True          # at C
    _git(tmp_path, "checkout", "-q", commits["B"])
    assert hz.head_is_phase3b_or_descendant(tmp_path) is True          # at tag
    _git(tmp_path, "checkout", "-q", commits["A"])
    assert hz.head_is_phase3b_or_descendant(tmp_path) is False         # before tag


def test_tag_missing_means_not_exists_and_not_descendant(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@e.com")
    _git(tmp_path, "config", "user.name", "T")
    (tmp_path / "f.txt").write_text("x\n", encoding="utf-8")
    _git(tmp_path, "add", "."); _git(tmp_path, "commit", "-q", "-m", "only")
    assert hz.phase3b_tag_exists(tmp_path) is False
    assert hz.head_is_phase3b_or_descendant(tmp_path) is False


def test_guard_refuses_phase3b_eval_when_tag_missing(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@e.com")
    _git(tmp_path, "config", "user.name", "T")
    (tmp_path / "f.txt").write_text("x\n", encoding="utf-8")
    _git(tmp_path, "add", "."); _git(tmp_path, "commit", "-q", "-m", "only")
    row = {"receipt_set": "phase3b_eval"}
    with pytest.raises(hz.EvalGuardError, match=f"the git tag '{hz.PHASE3B_TAG}' was not found"):
        hz.guard_phase3b_eval_run(row, repo_root=tmp_path)


def test_guard_passes_at_tag_and_descendant(tmp_path):
    commits = _build_phase3b_repo(tmp_path)  # HEAD at C (descendant)
    row = {"receipt_set": "phase3b_eval"}
    hz.guard_phase3b_eval_run(row, repo_root=tmp_path)             # must not raise
    _git(tmp_path, "checkout", "-q", commits["B"])
    hz.guard_phase3b_eval_run(row, repo_root=tmp_path)             # at tag: ok


def test_guard_refuses_before_tag_and_on_diverged_branch(tmp_path):
    commits = _build_phase3b_repo(tmp_path)
    row = {"receipt_set": "phase3b_eval"}
    _git(tmp_path, "checkout", "-q", commits["A"])
    with pytest.raises(hz.EvalGuardError, match="is not 'phase3b-prereg'"):
        hz.guard_phase3b_eval_run(row, repo_root=tmp_path)
    _git(tmp_path, "checkout", "-q", "-b", "diverged")
    (tmp_path / "f.txt").write_text("d\n", encoding="utf-8")
    _git(tmp_path, "add", "."); _git(tmp_path, "commit", "-q", "-m", "D")
    with pytest.raises(hz.EvalGuardError):
        hz.guard_phase3b_eval_run(row, repo_root=tmp_path)


# ---- start_run dirty-tree guard extended to phase3b_eval ----

def _init_repo(path: Path, dirty: bool) -> None:
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "t@e.com")
    _git(path, "config", "user.name", "T")
    (path / "file.txt").write_text("hello\n", encoding="utf-8")
    _git(path, "add", "."); _git(path, "commit", "-q", "-m", "init")
    if dirty:
        (path / "file.txt").write_text("changed\n", encoding="utf-8")


def _config():
    return hz.HarnessConfig(
        model_id="test/model",
        model_revision="0" * 40,
        max_new_tokens=16,
    )


def test_start_run_refuses_phase3b_eval_on_dirty_repo_before_writing(tmp_path):
    repo = tmp_path / "repo"; repo.mkdir()
    _init_repo(repo, dirty=True)
    out_dir = tmp_path / "out"
    with pytest.raises(hz.DirtyRepoEvalGuardError):
        hz.start_run(_config(), out_dir, receipt_set="phase3b_eval", repo_root=repo)
    assert not (out_dir / "environment.json").exists()


def test_start_run_allows_phase3b_eval_on_clean_repo(tmp_path):
    repo = tmp_path / "repo"; repo.mkdir()
    _init_repo(repo, dirty=False)
    out_dir = tmp_path / "out"
    info = hz.start_run(_config(), out_dir, receipt_set="phase3b_eval", repo_root=repo)
    assert info["git_dirty"] is False
    assert (out_dir / "environment.json").exists()


def test_start_run_still_allows_tune_on_dirty_repo(tmp_path):
    repo = tmp_path / "repo"; repo.mkdir()
    _init_repo(repo, dirty=True)
    out_dir = tmp_path / "out"
    hz.start_run(_config(), out_dir, receipt_set="tune", repo_root=repo)  # must not raise
    assert (out_dir / "environment.json").exists()
