//! Library root — re-exports the public API surface (port of `__init__.py`).

pub mod alerts;
pub mod api;
pub mod batch;
pub mod bayesian;
pub mod diagnose;
pub mod evidence;
pub mod history;
pub mod likelihoods;
pub mod reports;
pub mod sensitivity;

pub const VERSION: &str = "2.0.0";

pub use alerts::{AlertManager, AlertRule, Direction, ThresholdAlert};
pub use batch::BatchProcessor;
pub use bayesian::{BayesianDiagnostic, DiagnosisResult};
pub use diagnose::{
    diagnose, diagnose_interactive, diagnose_preset, find_preset, ScenarioPreset, PRESETS,
};
pub use evidence::{
    evidence_description, evidence_from_map, evidence_short_name, Evidence, EVIDENCE_PARAMS,
};
pub use history::{DiagnosisHistory, HistoryRecord};
pub use likelihoods::{
    default_likelihoods, failure_class_label, LikelihoodTable, DEFAULT_PRIORS,
    FAILURE_CLASSES, MILITARY_PRIORS,
};
pub use reports::{recommendations_for, ReportGenerator};
pub use sensitivity::{compute_sensitivity, FlipEntry, SensitivityReport};
