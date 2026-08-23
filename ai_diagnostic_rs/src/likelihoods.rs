//! Likelihood tables: P(e_i | Cause) for the five failure classes.

use crate::evidence::EVIDENCE_PARAMS;
use serde::Serialize;

pub const FAILURE_CLASSES: [&str; 5] = [
    "entropy",
    "engineering_limits",
    "human_error",
    "human_bias",
    "human_malice",
];

pub fn failure_class_label(cause: &str) -> &str {
    match cause {
        "entropy" => "Entropy (physical failure)",
        "engineering_limits" => "Engineering Limits (complexity)",
        "human_error" => "Human Error (unintentional)",
        "human_bias" => "Human Bias (systematic)",
        "human_malice" => "Human Malice (deliberate)",
        other => other,
    }
}

/// Default priors P(Cause), in canonical class order.
pub static DEFAULT_PRIORS: [(&str, f64); 5] = [
    ("entropy", 0.10),
    ("engineering_limits", 0.25),
    ("human_error", 0.25),
    ("human_bias", 0.25),
    ("human_malice", 0.15),
];

/// Military context priors (normalized to sum to 1.0; same ratios as the
/// paper's unnormalized values, which is what Bayes' theorem actually uses).
pub static MILITARY_PRIORS: [(&str, f64); 5] = [
    ("entropy", 0.01 / 0.31),
    ("engineering_limits", 0.05 / 0.31),
    ("human_error", 0.05 / 0.31),
    ("human_bias", 0.05 / 0.31),
    ("human_malice", 0.15 / 0.31),
];

pub const fn default_likelihoods() -> [[( &'static str, f64); 15]; 5] {
    [
        // entropy
        [
            ("e1_hardware_anomalies", 0.95),
            ("e2_data_distribution_shift", 0.05),
            ("e3_model_architecture_irregularities", 0.05),
            ("e4_human_interface_errors", 0.05),
            ("e5_temporal_pattern_sudden", 0.90),
            ("e6_output_coherence_adversarial", 0.05),
            ("e7_system_log_errors", 0.90),
            ("e8_external_environment_changes", 0.05),
            ("e9_obfuscated_code_or_weights", 0.01),
            ("e10_hidden_triggers_or_backdoors", 0.01),
            ("e11_unexplained_communication_channels", 0.01),
            ("e12_behavioral_context_inconsistency", 0.01),
            ("e13_unusual_training_data", 0.01),
            ("e14_designer_history_red_flags", 0.01),
            ("e15_legal_or_contractual_violations", 0.01),
        ],
        // engineering_limits
        [
            ("e1_hardware_anomalies", 0.05),
            ("e2_data_distribution_shift", 0.90),
            ("e3_model_architecture_irregularities", 0.90),
            ("e4_human_interface_errors", 0.05),
            ("e5_temporal_pattern_sudden", 0.40),
            ("e6_output_coherence_adversarial", 0.80),
            ("e7_system_log_errors", 0.50),
            ("e8_external_environment_changes", 0.10),
            ("e9_obfuscated_code_or_weights", 0.05),
            ("e10_hidden_triggers_or_backdoors", 0.05),
            ("e11_unexplained_communication_channels", 0.05),
            ("e12_behavioral_context_inconsistency", 0.10),
            ("e13_unusual_training_data", 0.10),
            ("e14_designer_history_red_flags", 0.05),
            ("e15_legal_or_contractual_violations", 0.05),
        ],
        // human_error
        [
            ("e1_hardware_anomalies", 0.05),
            ("e2_data_distribution_shift", 0.40),
            ("e3_model_architecture_irregularities", 0.30),
            ("e4_human_interface_errors", 0.95),
            ("e5_temporal_pattern_sudden", 0.30),
            ("e6_output_coherence_adversarial", 0.20),
            ("e7_system_log_errors", 0.40),
            ("e8_external_environment_changes", 0.80),
            ("e9_obfuscated_code_or_weights", 0.02),
            ("e10_hidden_triggers_or_backdoors", 0.02),
            ("e11_unexplained_communication_channels", 0.01),
            ("e12_behavioral_context_inconsistency", 0.05),
            ("e13_unusual_training_data", 0.05),
            ("e14_designer_history_red_flags", 0.02),
            ("e15_legal_or_contractual_violations", 0.02),
        ],
        // human_bias
        [
            ("e1_hardware_anomalies", 0.03),
            ("e2_data_distribution_shift", 0.40),
            ("e3_model_architecture_irregularities", 0.30),
            ("e4_human_interface_errors", 0.50),
            ("e5_temporal_pattern_sudden", 0.20),
            ("e6_output_coherence_adversarial", 0.80),
            ("e7_system_log_errors", 0.10),
            ("e8_external_environment_changes", 0.40),
            ("e9_obfuscated_code_or_weights", 0.03),
            ("e10_hidden_triggers_or_backdoors", 0.03),
            ("e11_unexplained_communication_channels", 0.02),
            ("e12_behavioral_context_inconsistency", 0.08),
            ("e13_unusual_training_data", 0.08),
            ("e14_designer_history_red_flags", 0.03),
            ("e15_legal_or_contractual_violations", 0.03),
        ],
        // human_malice
        [
            ("e1_hardware_anomalies", 0.01),
            ("e2_data_distribution_shift", 0.05),
            ("e3_model_architecture_irregularities", 0.05),
            ("e4_human_interface_errors", 0.05),
            ("e5_temporal_pattern_sudden", 0.40),
            ("e6_output_coherence_adversarial", 0.60),
            ("e7_system_log_errors", 0.10),
            ("e8_external_environment_changes", 0.05),
            ("e9_obfuscated_code_or_weights", 0.95),
            ("e10_hidden_triggers_or_backdoors", 0.95),
            ("e11_unexplained_communication_channels", 0.90),
            ("e12_behavioral_context_inconsistency", 0.90),
            ("e13_unusual_training_data", 0.90),
            ("e14_designer_history_red_flags", 0.80),
            ("e15_legal_or_contractual_violations", 0.80),
        ],
    ]
}

fn cause_index(cause: &str) -> Option<usize> {
    FAILURE_CLASSES.iter().position(|c| *c == cause)
}

fn evid_index(name: &str) -> Option<usize> {
    EVIDENCE_PARAMS.iter().position(|n| *n == name)
}

#[derive(Debug, Clone, Serialize)]
pub struct LikelihoodTable {
    /// P(Cause) per class, ordered by FAILURE_CLASSES.
    pub priors: Vec<(&'static str, f64)>,
    /// P(e_i | Cause): one row per cause, each row ordered by EVIDENCE_PARAMS.
    pub likelihoods: Vec<(&'static str, Vec<(&'static str, f64)>)>,
}

impl Default for LikelihoodTable {
    fn default() -> Self {
        Self::new()
    }
}

impl LikelihoodTable {
    pub fn new() -> Self {
        LikelihoodTable {
            priors: DEFAULT_PRIORS.to_vec(),
            likelihoods: FAILURE_CLASSES
                .iter()
                .zip(default_likelihoods())
                .map(|(cause, row)| (*cause, row.to_vec()))
                .collect(),
        }
    }

    pub fn with_priors(priors: &[(&'static str, f64)]) -> Self {
        let mut table = Self::new();
        for (cause, val) in priors {
            table
                .set_prior(cause, *val)
                .expect("valid cause and range");
        }
        table
    }

    pub fn prior(&self, cause: &str) -> Option<f64> {
        self.priors.iter().find(|(c, _)| *c == cause).map(|(_, v)| *v)
    }

    pub fn set_prior(&mut self, cause: &str, value: f64) -> Result<(), String> {
        if !(0.0..=1.0).contains(&value) {
            return Err(format!("Prior must be in [0, 1], got {value}"));
        }
        match self.priors.iter_mut().find(|(c, _)| *c == cause) {
            Some((_, v)) => {
                *v = value;
                Ok(())
            }
            None => Err(format!("Unknown cause: {cause}")),
        }
    }

    pub fn p_evidence_given_cause(&self, cause: &str, evidence_name: &str) -> Option<f64> {
        let ci = cause_index(cause)?;
        let ei = evid_index(evidence_name)?;
        self.likelihoods.get(ci)?.1.get(ei).map(|(_, v)| *v)
    }

    pub fn set_likelihood(
        &mut self,
        cause: &str,
        evidence_name: &str,
        value: f64,
    ) -> Result<(), String> {
        if !(0.0..=1.0).contains(&value) {
            return Err(format!("Likelihood must be in [0, 1], got {value}"));
        }
        let ci = cause_index(cause).ok_or_else(|| format!("Unknown cause: {cause}"))?;
        let ei =
            evid_index(evidence_name).ok_or_else(|| format!("Unknown evidence: {evidence_name}"))?;
        self.likelihoods[ci].1[ei].1 = value;
        Ok(())
    }

    pub fn validate(&self) -> Vec<String> {
        let mut warnings = Vec::new();
        let prior_sum: f64 = self.priors.iter().map(|(_, v)| v).sum();
        if (prior_sum - 1.0).abs() > 0.01 {
            warnings.push(format!(
                "Priors sum to {prior_sum:.4}, expected 1.0"
            ));
        }
        for cause in FAILURE_CLASSES {
            for evid in EVIDENCE_PARAMS {
                if self.p_evidence_given_cause(cause, evid) == Some(0.0) {
                    warnings.push(format!(
                        "P({evid} | {cause}) = 0.0 — will zero out entire posterior. \
                         Consider using a small floor value (e.g., 0.01)."
                    ));
                }
            }
        }
        warnings
    }
}
