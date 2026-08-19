"""Loads and queries the verified checkpoint registry (configs/models/*.yaml)."""

from __future__ import annotations

from pathlib import Path

import yaml

from mpe.checkpoints.schema import CheckpointSpec, CheckpointStage, LineageSpec


class CheckpointRegistry:
    """Single source of truth for "what does Base/SFT/DPO/RLVR resolve to".

    Deliberately does not hit the network or validate that the Hub repos
    still exist — this class only parses and serves the registry file that
    Phase 0/1 verification already produced. Re-verifying against the live
    Hub is a separate, explicit step (see docs/phase0-findings.md), not
    something that should happen implicitly every time a config is loaded.
    """

    def __init__(self, lineages: dict[str, LineageSpec], selected_for_mvp: str | None = None):
        self._lineages = lineages
        self.selected_for_mvp = selected_for_mvp

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CheckpointRegistry":
        path = Path(path)
        raw = yaml.safe_load(path.read_text())
        lineages_raw = raw.get("lineages", {})
        lineages: dict[str, LineageSpec] = {}
        for name, lineage_data in lineages_raw.items():
            lineages[name] = LineageSpec(name=name, **lineage_data)
        if not lineages:
            raise ValueError(f"No lineages found in {path}")
        return cls(lineages=lineages, selected_for_mvp=raw.get("selected_for_mvp"))

    @property
    def lineage_names(self) -> list[str]:
        return list(self._lineages.keys())

    def get_lineage(self, name: str) -> LineageSpec:
        try:
            return self._lineages[name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown lineage '{name}'. Known lineages: {self.lineage_names}"
            ) from exc

    def get_checkpoint(self, lineage: str, stage: CheckpointStage) -> CheckpointSpec:
        return self.get_lineage(lineage).get_stage(stage)

    def mvp_lineage(self) -> LineageSpec:
        if self.selected_for_mvp is None:
            raise ValueError("Registry does not declare a selected_for_mvp lineage")
        return self.get_lineage(self.selected_for_mvp)
