"""
AI Control Failure Diagnostic Framework
=======================================

A probabilistic diagnostic tool for classifying AI anomalous behavior
into one of five failure classes using Bayesian inference over 15
observable evidence parameters.

Based on: "Human versus AI: The Gödelian Referee" (Barii, 2026)
"""

__version__ = "2.0.0"

from .evidence import Evidence, evidence_from_dict, evidence_vector_names, EVIDENCE_PARAMS, EVIDENCE_DESCRIPTIONS
from .likelihoods import LikelihoodTable, FAILURE_CLASSES, FAILURE_CLASS_LABELS
from .bayesian import BayesianDiagnostic, DiagnosisResult
from .diagnose import diagnose, diagnose_preset, diagnose_interactive, PRESETS
from .sensitivity import SensitivityReport, compute_sensitivity
from .history import DiagnosisHistory
from .batch import BatchProcessor
from .alerts import AlertManager, ThresholdAlert
from .reports import ReportGenerator

__all__ = [
    "Evidence",
    "evidence_from_dict",
    "evidence_vector_names",
    "EVIDENCE_PARAMS",
    "EVIDENCE_DESCRIPTIONS",
    "LikelihoodTable",
    "FAILURE_CLASSES",
    "FAILURE_CLASS_LABELS",
    "BayesianDiagnostic",
    "DiagnosisResult",
    "diagnose",
    "diagnose_preset",
    "diagnose_interactive",
    "PRESETS",
    "SensitivityReport",
    "compute_sensitivity",
    "DiagnosisHistory",
    "BatchProcessor",
    "AlertManager",
    "ThresholdAlert",
    "ReportGenerator",
]
