"""
MLflow tracking utilities for TriadRank ATS.
Centralised configuration and helpers for experiment tracking,
parameter logging, metric logging, and artifact logging.
"""
from __future__ import annotations

import os
import logging
from typing import Any, Dict, Optional
from pathlib import Path

import mlflow

logger = logging.getLogger(__name__)


def setup_experiment(
    config: Dict[str, Any],
    experiment_name: str,
    tracking_uri: Optional[str] = None,
) -> str:
    """Configure the MLflow tracking URI and create / set the named experiment.

    Args:
        config: The project config dict (must contain an optional "mlflow" key).
        experiment_name: Name for the experiment (e.g. "cross-encoder").
        tracking_uri: Override URI; defaults to *config.mlflow.tracking_uri*.

    Returns:
        The experiment ID (so callers can pass it to ``start_run``).
    """
    mlflow_cfg = config.get("mlflow", {})
    uri = tracking_uri or mlflow_cfg.get("tracking_uri", "mlruns")
    mlflow.set_tracking_uri(uri)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        exp_id = mlflow.create_experiment(
            experiment_name,
            artifact_location=mlflow_cfg.get("artifact_location"),
        )
    else:
        exp_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)

    if mlflow_cfg.get("pytorch_autolog", False):
        mlflow.pytorch.autolog()

    logger.info("MLflow experiment '%s' ready (tracking URI: %s)", experiment_name, uri)
    return exp_id


def log_params_from_config(
    config: Dict[str, Any],
    key_path: str,
    *,
    run_id: Optional[str] = None,
) -> None:
    """Log a flat dict of parameters from a nested config key path.

    Example::

        log_params_from_config(config, "models.cross_encoder")

    logs ``learning_rate=2e-5``, ``batch_size=16``, ``dropout=0.1``, etc.
    as MLflow parameters.  Supports ``max_seq_length``, ``fp16`` bools,
    ``gradient_accumulation_steps`` ints — any leaf value that is a str,
    int, float, or bool.
    """
    parts = key_path.split(".")
    obj: Any = config
    for part in parts:
        if isinstance(obj, dict):
            obj = obj.get(part, {})
        else:
            logger.warning("Key path '%s' could not be resolved at '%s'", key_path, part)
            return

    if not isinstance(obj, dict):
        logger.warning("Key path '%s' does not point to a dict", key_path)
        return

    params = {}
    for k, v in obj.items():
        if isinstance(v, (str, int, float, bool)):
            params[k] = str(v)

    if run_id:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_params(params)
    else:
        mlflow.log_params(params)


def log_training_metrics(
    metrics: Dict[str, float],
    step: Optional[int] = None,
    *,
    run_id: Optional[str] = None,
) -> None:
    """Log a flat dict of scalar metrics at an optional step.

    Convenience wrapper that filters to numeric values.
    """
    numeric = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
    if run_id:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(numeric, step=step)
    else:
        mlflow.log_metrics(numeric, step=step)


def log_artifacts(
    local_dir: str,
    artifact_path: Optional[str] = None,
    *,
    run_id: Optional[str] = None,
) -> None:
    """Log a directory of artifacts to the current (or named) MLflow run."""
    if run_id:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifacts(local_dir, artifact_path=artifact_path)
    else:
        mlflow.log_artifacts(local_dir, artifact_path=artifact_path)


def register_model(
    model_uri: str,
    model_name: str,
    stage: str = "None",
) -> Optional[mlflow.entities.model_registry.ModelVersion]:
    """Register a model in the MLflow Model Registry.

    Args:
        model_uri: URI of the logged model (e.g. ``runs:/<run_id>/model``).
        model_name: Registered model name.
        stage: Target stage — ``"None"``, ``"Staging"``, ``"Production"``, or ``"Archived"``.

    Returns:
        The created ModelVersion, or ``None`` if registration failed.
    """
    try:
        result = mlflow.register_model(model_uri, model_name)
        if stage and stage != "None":
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(model_name, result.version, stage)
        logger.info("Model '%s' version %s registered as '%s'", model_name, result.version, stage)
        return result
    except Exception as exc:
        logger.warning("Model registration failed: %s", exc)
        return None
