//! Bayesian diagnostic engine: computes P(Cause | evidence).

use crate::evidence::{Evidence, EVIDENCE_PARAMS};
use crate::likelihoods::{failure_class_label, LikelihoodTable, FAILURE_CLASSES};
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::fmt;

const LOG_FLOOR: f64 = 1e-300;

#[derive(Debug, Clone)]
pub struct DiagnosisResult {
    /// Posterior per cause, ordered by FAILURE_CLASSES.
    pub posteriors: Vec<(String, f64)>,
    pub evidence_vector: Evidence,
    pub diagnosis: String,
    pub confidence: f64,
    pub needs_investigation: bool,
    /// Log P(e | Cause) + log prior per cause, ordered by FAILURE_CLASSES.
    pub log_likelihoods: Vec<(String, f64)>,
}

impl DiagnosisResult {
    pub fn posterior(&self, cause: &str) -> f64 {
        self.posteriors
            .iter()
            .find(|(c, _)| c == cause)
            .map(|(_, v)| *v)
            .unwrap_or(0.0)
    }

    fn sorted_posteriors(&self) -> Vec<(&str, f64)> {
        let mut pairs: Vec<(&str, f64)> = self
            .posteriors
            .iter()
            .map(|(c, v)| (c.as_str(), *v))
            .collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        pairs
    }
}

impl fmt::Display for DiagnosisResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut lines = vec![
            "=".repeat(60),
            "  AI CONTROL FAILURE DIAGNOSIS".to_string(),
            "=".repeat(60),
            String::new(),
            "Evidence present:".to_string(),
            self.evidence_vector.summary(),
            String::new(),
            "-".repeat(60),
            "  Posterior Probabilities".to_string(),
            "-".repeat(60),
        ];
        for (cause, prob) in self.sorted_posteriors() {
            let label = failure_class_label(cause);
            let bar = "#".repeat((prob * 40.0) as usize);
            lines.push(format!("  {label:<35} {prob:>8.4}  {bar}"));
        }
        lines.push(String::new());
        lines.push("-".repeat(60));
        lines.push(format!(
            "  DIAGNOSIS: {}",
            failure_class_label(&self.diagnosis)
        ));
        lines.push(format!("  CONFIDENCE: {:.2}%", self.confidence * 100.0));
        if self.needs_investigation {
            lines.push(
                "  ⚠ Confidence below threshold — further investigation recommended."
                    .to_string(),
            );
        }
        lines.push("=".repeat(60));
        write!(f, "{}", lines.join("\n"))
    }
}

impl DiagnosisResult {
    pub fn to_dict(&self) -> Map<String, Value> {
        let mut m = Map::new();
        m.insert(
            "posteriors".into(),
            Value::Object(
                self.posteriors
                    .iter()
                    .map(|(c, v)| (c.clone(), json!(v)))
                    .collect(),
            ),
        );
        m.insert("diagnosis".into(), json!(self.diagnosis));
        m.insert(
            "diagnosis_label".into(),
            json!(failure_class_label(&self.diagnosis)),
        );
        m.insert("confidence".into(), json!(self.confidence));
        m.insert("needs_investigation".into(), json!(self.needs_investigation));
        m.insert(
            "log_likelihoods".into(),
            Value::Object(
                self.log_likelihoods
                    .iter()
                    .map(|(c, v)| (c.clone(), json!(v)))
                    .collect(),
            ),
        );
        m
    }

    pub fn to_json_string(&self) -> String {
        Value::Object(self.to_dict()).to_string()
    }
}

#[derive(Debug, Clone, Default)]
pub struct BayesianDiagnostic {
    pub table: LikelihoodTable,
}

impl BayesianDiagnostic {
    pub fn new() -> Self {
        BayesianDiagnostic {
            table: LikelihoodTable::new(),
        }
    }

    pub fn with_table(table: LikelihoodTable) -> Self {
        BayesianDiagnostic { table }
    }

    /// Compute posteriors over the five failure classes.
    ///
    /// `prior_overrides` selectively replaces P(Cause) values.
    pub fn diagnose(
        &self,
        evidence: &Evidence,
        prior_overrides: Option<&HashMap<String, f64>>,
        confidence_threshold: f64,
    ) -> Result<DiagnosisResult, String> {
        evidence.validate()?;

        let log_likelihoods = self.compute_log_joint(evidence, prior_overrides);

        let max_log = log_likelihoods
            .iter()
            .map(|(_, v)| *v)
            .fold(f64::NEG_INFINITY, f64::max);
        let exp_shifted: Vec<(String, f64)> = log_likelihoods
            .iter()
            .map(|(c, v)| (c.clone(), (v - max_log).exp()))
            .collect();
        let total: f64 = exp_shifted.iter().map(|(_, v)| v).sum();
        let posteriors: Vec<(String, f64)> = exp_shifted
            .into_iter()
            .map(|(c, v)| (c, v / total))
            .collect();

        let (diagnosis, confidence) = posteriors
            .iter()
            .fold(("", f64::MIN), |acc, (c, v)| if *v > acc.1 { (c.as_str(), *v) } else { acc });
        let diagnosis = diagnosis.to_string();
        let needs_investigation = confidence < confidence_threshold;

        Ok(DiagnosisResult {
            posteriors,
            evidence_vector: evidence.clone(),
            diagnosis,
            confidence,
            needs_investigation,
            log_likelihoods,
        })
    }

    /// log P(e | C) + log P(C) per cause; overrides applied to a cloned table.
    fn compute_log_joint(
        &self,
        evidence: &Evidence,
        prior_overrides: Option<&HashMap<String, f64>>,
    ) -> Vec<(String, f64)> {
        let ev_vec = evidence.to_vector();

        FAILURE_CLASSES
            .iter()
            .map(|cause| {
                let base_prior = self.table.prior(cause).unwrap_or(LOG_FLOOR);
                let prior = prior_overrides
                    .and_then(|ov| ov.get(*cause).copied())
                    .unwrap_or(base_prior);
                let log_prior = prior.max(LOG_FLOOR).ln();

                let log_lik: f64 = EVIDENCE_PARAMS
                    .iter()
                    .zip(ev_vec.iter())
                    .map(|(name, ei)| {
                        let pe_c = self
                            .table
                            .p_evidence_given_cause(cause, name)
                            .unwrap_or(0.0);
                        if *ei == 0.0 {
                            (1.0 - pe_c).max(LOG_FLOOR).ln()
                        } else if *ei == 1.0 {
                            pe_c.max(LOG_FLOOR).ln()
                        } else {
                            ei * pe_c.max(LOG_FLOOR).ln()
                                + (1.0 - ei) * (1.0 - pe_c).max(LOG_FLOOR).ln()
                        }
                    })
                    .sum();

                ((*cause).to_string(), log_prior + log_lik)
            })
            .collect()
    }
}
