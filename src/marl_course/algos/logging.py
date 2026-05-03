"""Metric logging utilities (CSV + optional Weights & Biases)."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class MetricLogger:
    """CSV logger with optional Weights & Biases mirroring."""

    def __init__(
        self,
        out_dir: Path,
        csv_name: str = "metrics.csv",
        use_wandb: bool = False,
        project: str = "marl-course-games",
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
    ):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.out_dir / csv_name
        self.file = self.csv_path.open("w", newline="", encoding="utf-8")
        self.writer: csv.DictWriter[str] | None = None
        self.wandb = None
        if use_wandb:
            try:
                import wandb

                self.wandb = wandb
                # The full effective JSON config is mirrored into W&B so runs
                # can be reproduced without checking shell history.
                wandb.init(project=project, name=run_name, config=config or {})
            except Exception as exc:
                print(f"W&B disabled: {exc}")
                self.wandb = None

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Append one metric row to CSV and mirror to W&B if enabled."""
        row = dict(metrics)
        if step is not None:
            row.setdefault("step", step)
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=list(row.keys()))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.file.flush()
        if self.wandb is not None:
            self.wandb.log(metrics, step=step)

    def log_artifact_file(self, path: Path, name: str, artifact_type: str = "artifact") -> None:
        """Upload a file to W&B when enabled; otherwise keep the local file."""

        if self.wandb is None or not path.exists():
            return
        artifact = self.wandb.Artifact(name, type=artifact_type)
        artifact.add_file(str(path))
        self.wandb.log_artifact(artifact)

    def log_gif(self, path: Path, key: str = "post_training_episode", fps: int = 8) -> None:
        """Log an episode GIF to W&B as media and as a downloadable artifact."""

        if self.wandb is None or not path.exists():
            return
        self.wandb.log({key: self.wandb.Video(str(path), fps=fps, format="gif")})
        self.log_artifact_file(path, name=key, artifact_type="episode-gif")

    def close(self) -> None:
        """Close file handles and finalize remote run."""
        self.file.close()
        if self.wandb is not None:
            self.wandb.finish()
