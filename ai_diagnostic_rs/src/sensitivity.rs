//! Sensitivity analysis: parameter impact, flip analysis, tipping points,
//! prior perturbation.

use crate::bayesian::{BayesianDiagnostic, DiagnosisResult};
use crate::evidence::EVIDENCE_PARAMS;
use crate::likelihoods::FAILURE_CLASSES;
use serde_json::{json, Map, Value};

#[derive(Debug, Clone, serde::Serialize)]
pub struct FlipEntry {
    pub parameter: String,
    pub current_value: f64,
    pub alternative_diagnosis: String,
    pub new_confidence: f64,
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct SensitivityReport {
    pub parameter_sensitivity: Vec<(String, f64)>,
    pub flip_analysis: Vec<FlipEntry>,
    pub tipping_points: Vec<(String, Option<f64>)>,
    pub prior_sensitivity: Vec<(String, f64)>,
}

impl SensitivityReport {
    fn sorted<'a, I>(items: I) -> Vec<I::Item>
    where
        I: IntoIterator<Item = (&'a str, f64)>,
    {
        let mut v: Vec<_> = items.into_iter().collect();
        v.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal));
        v
    }

    pub fn summary(&self) -> String {
        let mut lines = vec![
            "=".repeat(60),
            "  SENSITIVITY ANALYSIS".to_string(),
            "=".repeat(60),
            String::new(),
            "-".repeat(60),
            "  Parameter Sensitivity (|delta in top posterior|)".to_string(),
            "-".repeat(60),
        ];
        if self.parameter_sensitivity.is_empty() {
            lines.push("  (none)".to_string());
        } else {
            for (name, delta) in Self::sorted(
                self.parameter_sensitivity
                    .iter()
                    .map(|(n, d)| (n.as_str(), *d)),
            ) {
                lines.push(format!("  {name:<50} {delta:>+8.4}"));
            }
        }

        lines.push(String::new());
        lines.push("-".repeat(60));
        lines.push(
            "  Flip Analysis (missing evidence that would change diagnosis)".to_string(),
        );
        lines.push("-".repeat(60));
        if self.flip_analysis.is_empty() {
            lines.push("  (no missing evidence that would flip diagnosis)".to_string());
        } else {
            for e in &self.flip_analysis {
                lines.push(format!(
                    "  {:<40} → {:<25} conf={:.2}",
                    e.parameter,
                    e.alternative_diagnosis,
                    e.new_confidence * 100.0
                ));
            }
        }

        lines.push(String::new());
        lines.push("-".repeat(60));
        lines.push("  Tipping Points (evidence value at which diagnosis flips)".to_string());
        lines.push("-".repeat(60));
        if self.tipping_points.is_empty() {
            lines.push("  (none)".to_string());
        } else {
            for (name, val) in &self.tipping_points {
                match val {
                    None => lines.push(format!("  {name:<50}   (no flip in [0, 1])")),
                    Some(v) => lines.push(format!("  {name:<50}   {v:.4}")),
                }
            }
        }

        lines.push(String::new());
        lines.push("-".repeat(60));
        lines.push("  Prior Sensitivity (|delta| for ±10% perturbation)".to_string());
        lines.push("-".repeat(60));
        if self.prior_sensitivity.is_empty() {
            lines.push("  (none)".to_string());
        } else {
            for (cause, delta) in Self::sorted(
                self.prior_sensitivity.iter().map(|(c, d)| (c.as_str(), *d)),
            ) {
                lines.push(format!("  {cause:<50} {delta:>+8.4}"));
            }
        }

        lines.push("=".repeat(60));
        lines.join("\n")
    }

    pub fn to_dict(&self) -> Map<String, Value> {
        let mut m = Map::new();
        m.insert(
            "parameter_sensitivity".into(),
            Value::Object(
                self.parameter_sensitivity
                    .iter()
                    .map(|(k, v)| (k.clone(), json!(v)))
                    .collect(),
            ),
        );
        m.insert(
            "flip_analysis".into(),
            json!(self.flip_analysis),
        );
        m.insert(
            "tipping_points".into(),
            Value::Object(
                self.tipping_points
                    .iter()
                    .map(|(k, v)| (k.clone(), json!(v)))
                    .collect(),
            ),
        );
        m.insert(
            "prior_sensitivity".into(),
            Value::Object(
                self.prior_sensitivity
                    .iter()
                    .map(|(k, v)| (k.clone(), json!(v)))
                    .collect(),
            ),
        );
        m
    }
}

fn top_of(result: &DiagnosisResult) -> f64 {
    result.posteriors
        .iter()
        .find(|(c, _)| *c == result.diagnosis)
        .map(|(_, v)| *v)
        .unwrap_or(0.0)
}

/// Parameter sensitivity: zero each present param, measure delta in top posterior.
fn compute_parameter_sensitivity(
    result: &DiagnosisResult,
    diagnostic: &BayesianDiagnostic,
) -> Result<Vec<(String, f64)>, String> {
    let base_confidence = result.confidence;
    let evidence = &result.evidence_vector;
    let mut out = Vec::new();

    for param in EVIDENCE_PARAMS {
        if evidence.get(param).unwrap() == 0.0 {
            out.push((param.to_string(), 0.0));
            continue;
        }
        let mut ev = evidence.clone();
        ev.set(param, 0.0)?;
        let new_result = diagnostic.diagnose(&ev, None, 0.6)?;
        out.push((param.to_string(), top_of(&new_result) - base_confidence));
    }
    Ok(out)
}

/// Flip analysis: set each absent param to 1.0; record diagnosis changes.
fn compute_flip_analysis(
    result: &DiagnosisResult,
    diagnostic: &BayesianDiagnostic,
) -> Result<Vec<FlipEntry>, String> {
    let evidence = &result.evidence_vector;
    let current_diagnosis = &result.diagnosis;
    let mut flips = Vec::new();

    for param in EVIDENCE_PARAMS {
        if evidence.get(param).unwrap() != 0.0 {
            continue;
        }
        let mut ev = evidence.clone();
        ev.set(param, 1.0)?;
        let new_result = diagnostic.diagnose(&ev, None, 0.6)?;
        if &new_result.diagnosis != current_diagnosis {
            flips.push(FlipEntry {
                parameter: param.to_string(),
                current_value: 0.0,
                alternative_diagnosis: new_result.diagnosis.clone(),
                new_confidence: new_result.confidence,
            });
        }
    }
    Ok(flips)
}

/// Tipping points: bisect each param toward 1.0 to find flip threshold.
fn compute_tipping_points(
    result: &DiagnosisResult,
    diagnostic: &BayesianDiagnostic,
) -> Result<Vec<(String, Option<f64>)>, String> {
    let evidence = &result.evidence_vector;
    let top_diagnosis = &result.diagnosis;
    let mut tipping = Vec::new();

    for param in EVIDENCE_PARAMS {
        let current = evidence.get(param).unwrap();
        if current >= 1.0 {
            tipping.push((param.to_string(), None));
            continue;
        }
        let (mut lo, mut hi) = (current, 1.0f64);
        let mut found: Option<f64> = None;

        for _ in 0..50 {
            let mid = (lo + hi) / 2.0;
            let mut ev = evidence.clone();
            ev.set(param, mid)?;
            let probe = diagnostic.diagnose(&ev, None, 0.6)?;
            if probe.diagnosis != *top_diagnosis {
                found = Some(mid);
                hi = mid;
            } else {
                lo = mid;
            }
            if hi - lo < 1e-6 {
                break;
            }
        }
        tipping.push((param.to_string(), found));
    }
    Ok(tipping)
}

/// Prior sensitivity: perturb each prior +10% (capped at 1.0), measure delta.
fn compute_prior_sensitivity(
    result: &DiagnosisResult,
    diagnostic: &BayesianDiagnostic,
) -> Result<Vec<(String, f64)>, String> {
    let base_confidence = result.confidence;
    let evidence = &result.evidence_vector;
    let mut out = Vec::new();

    for cause in FAILURE_CLASSES {
        let base_prior = diagnostic.table.prior(cause).unwrap_or(0.0);
        let overrides = std::collections::HashMap::from([(
            cause.to_string(),
            (base_prior * 1.1).min(1.0),
        )]);
        let up_result = diagnostic.diagnose(evidence, Some(&overrides), 0.6)?;
        let delta = top_of(&up_result)
            - base_confidence;
        out.push((cause.to_string(), delta));
    }
    Ok(out)
}

/// Compute all four sensitivity analyses for a given diagnosis.
///
/// Sub-diagnoses use the engine's default table, mirroring the reference
/// implementation's behavior.
pub fn compute_sensitivity(
    result: &DiagnosisResult,
    diagnostic: &BayesianDiagnostic,
) -> Result<SensitivityReport, String> {
    Ok(SensitivityReport {
        parameter_sensitivity: compute_parameter_sensitivity(result, diagnostic)?,
        flip_analysis: compute_flip_analysis(result, diagnostic)?,
        tipping_points: compute_tipping_points(result, diagnostic)?,
        prior_sensitivity: compute_prior_sensitivity(result, diagnostic)?,
    })
}
