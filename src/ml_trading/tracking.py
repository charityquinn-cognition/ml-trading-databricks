"""MLflow logging for a completed experiment.

MLflow is an optional dependency: importing this module is only worthwhile inside
Databricks or with a tracking server configured, so nothing else in the package
imports it at module scope.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ml_trading.pipeline import ExperimentResult

logger = logging.getLogger(__name__)


def log_experiment(
    result: ExperimentResult,
    *,
    artifacts_dir: Path | None = None,
    experiment_name: str | None = None,
    run_name: str | None = None,
) -> str:
    """Log the config, metrics, per-fold diagnostics and artifacts of one run.

    Returns the MLflow run id so a caller can register or compare runs.
    """
    import mlflow

    if experiment_name:
        mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name or result.config.name) as run:
        mlflow.log_params(_stringify(result.config.flat_params()))
        mlflow.log_metrics(result.metrics())
        for fold in result.folds:
            mlflow.log_metrics(
                {
                    f"{fold.label}.auc": fold.auc,
                    f"{fold.label}.ic": fold.information_coefficient,
                    f"{fold.label}.train_rows": fold.train_rows,
                },
            )
        commit = _git_commit()
        if commit:
            mlflow.set_tag("git_commit", commit)
        mlflow.set_tag("model", result.config.model.name)
        mlflow.set_tag("label_horizon", result.config.label.horizon)
        if artifacts_dir is not None and Path(artifacts_dir).exists():
            mlflow.log_artifacts(str(artifacts_dir))
        return str(run.info.run_id)


def _stringify(params: dict[str, object]) -> dict[str, str]:
    return {key: str(value) for key, value in params.items()}


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):  # pragma: no cover
        return None
