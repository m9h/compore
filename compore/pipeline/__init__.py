"""
Production SBI Pipeline for dmipy-JAX.

Provides a unified train -> serialize -> deploy workflow for
simulation-based inference of diffusion MRI microstructure parameters.
"""

from compore.pipeline.config import SBIPipelineConfig
from compore.pipeline.simulator import ModelSimulator
from compore.pipeline.train import train_sbi
from compore.pipeline.checkpoint import save_checkpoint, load_checkpoint
from compore.pipeline.deploy import SBIPredictor
from compore.pipeline.ensemble import (
    train_ensemble,
    EnsemblePredictor,
    save_ensemble,
    load_ensemble,
)
from compore.pipeline.conformal import (
    ConformalCalibrator,
    ConformalResult,
    calibrate_from_predictions,
    predict_intervals_from_predictions,
)

__all__ = [
    "SBIPipelineConfig",
    "ModelSimulator",
    "train_sbi",
    "save_checkpoint",
    "load_checkpoint",
    "SBIPredictor",
    "train_ensemble",
    "EnsemblePredictor",
    "save_ensemble",
    "load_ensemble",
    "ConformalCalibrator",
    "ConformalResult",
    "calibrate_from_predictions",
    "predict_intervals_from_predictions",
]
