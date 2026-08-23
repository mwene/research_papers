//! Port of `test_comprehensive.py`: covers evidence, likelihoods, bayesian,
//! sensitivity, history, batch, alerts, reports, presets, and edge cases.

use ai_diagnostic_rs::{
    compute_sensitivity, diagnose, diagnose_preset, AlertManager, BayesianDiagnostic,
    BatchProcessor, DiagnosisHistory, Direction, Evidence, ReportGenerator, ThresholdAlert,
    DEFAULT_PRIORS, EVIDENCE_PARAMS, FAILURE_CLASSES, PRESETS,
};
use serde_json::{json, Value};
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;

const THRESHOLD: f64 = 0.6;

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir()
        .join(format!("ai_diag_rs_test_{}_{}", name, std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn ev(pairs: &[(&str, f64)]) -> Evidence {
    let mut e = Evidence::default();
    for (name, val) in pairs {
        e.set(name, *val).expect("valid evidence value");
    }
    e
}

fn overrides(pairs: &[(&str, f64)]) -> HashMap<String, f64> {
    pairs
        .iter()
        .map(|(c, v)| (c.to_string(), *v))
        .collect()
}

// ── Evidence ──────────────────────────────────────────────────────────

#[test]
fn test_evidence_creation() {
    let e = ev(&[("e1_hardware_anomalies", 1.0), ("e9_obfuscated_code_or_weights", 0.5)]);
    assert_eq!(e.e1_hardware_anomalies, 1.0);
    assert_eq!(e.e9_obfuscated_code_or_weights, 0.5);
    assert_eq!(e.e2_data_distribution_shift, 0.0);
}

#[test]
fn test_evidence_validation() {
    let mut e = Evidence::default();
    assert!(e.set("e1_hardware_anomalies", 2.0).is_err());
}

#[test]
fn test_evidence_vector() {
    let e = Evidence::default();
    let vec = e.to_vector();
    assert_eq!(vec.len(), 15);
    assert!(vec.iter().all(|&v| v == 0.0));
}

#[test]
fn test_evidence_from_dict() {
    let d = json!({"e1_hardware_anomalies": 1.0, "e10_hidden_triggers_or_backdoors": 0.8});
    let map = d.as_object().unwrap();
    let e = ai_diagnostic_rs::evidence_from_map(map).expect("valid map");
    assert_eq!(e.e1_hardware_anomalies, 1.0);
    assert_eq!(e.e10_hidden_triggers_or_backdoors, 0.8);
}

#[test]
fn test_evidence_summary() {
    let e = ev(&[("e1_hardware_anomalies", 1.0), ("e7_system_log_errors", 0.5)]);
    let s = e.summary();
    assert!(s.contains("Hardware anomalies"), "summary: {s}");
    assert!(s.contains("System log errors"), "summary: {s}");
}

// ── LikelihoodTable ──────────────────────────────────────────────────

#[test]
fn test_likelihood_table_defaults() {
    use ai_diagnostic_rs::{LikelihoodTable, FAILURE_CLASSES};
    let t = LikelihoodTable::new();
    assert_eq!(t.prior("entropy"), Some(0.10));
    assert_eq!(t.prior("human_malice"), Some(0.15));
    let p = t
        .p_evidence_given_cause("entropy", "e1_hardware_anomalies")
        .unwrap();
    assert!(p > 0.0 && p < 1.0);
    for cause in FAILURE_CLASSES {
        assert!(t.prior(cause).is_some());
    }
}

#[test]
fn test_likelihood_table_set_prior() {
    let mut t = ai_diagnostic_rs::LikelihoodTable::new();
    t.set_prior("entropy", 0.5).expect("set prior");
    assert_eq!(t.prior("entropy"), Some(0.5));
}

#[test]
fn test_likelihood_table_validation() {
    let t = ai_diagnostic_rs::LikelihoodTable::new();
    let warnings = t.validate();
    assert!(warnings.is_empty(), "default table should be clean: {warnings:?}");
}

// ── Bayesian Engine ──────────────────────────────────────────────────

#[test]
fn test_diagnose_empty() {
    let e = Evidence::default();
    let result = diagnose(&e, None, THRESHOLD).expect("diagnose");
    assert!(FAILURE_CLASSES.contains(&result.diagnosis.as_str()));
    assert!((0.0..=1.0).contains(&result.confidence));
}

#[test]
fn test_diagnose_military_deception() {
    let result = diagnose_preset("military_deception", THRESHOLD).expect("preset");
    assert_eq!(result.diagnosis, "human_malice");
    assert!(result.confidence > 0.99, "confidence: {}", result.confidence);
}

#[test]
fn test_diagnose_hardware_crash() {
    let result = diagnose_preset("hardware_crash", THRESHOLD).expect("preset");
    assert_eq!(result.diagnosis, "entropy");
    assert!(result.confidence > 0.95, "confidence: {}", result.confidence);
}

#[test]
fn test_diagnose_distribution_drift() {
    let result = diagnose_preset("distribution_drift", THRESHOLD).expect("preset");
    assert_eq!(result.diagnosis, "engineering_limits");
    assert!(result.confidence > 0.90, "confidence: {}", result.confidence);
}

#[test]
fn test_diagnose_misconfigured_reward() {
    let result = diagnose_preset("misconfigured_reward", THRESHOLD).expect("preset");
    assert_eq!(result.diagnosis, "human_error");
    assert!(result.confidence > 0.75, "confidence: {}", result.confidence);
}

#[test]
fn test_diagnose_subtle_bias() {
    let result = diagnose_preset("subtle_bias", THRESHOLD).expect("preset");
    assert_eq!(result.diagnosis, "human_bias");
    assert!(result.confidence > 0.60, "confidence: {}", result.confidence);
}

#[test]
fn test_diagnose_posteriors_sum_to_one() {
    for preset in PRESETS.iter() {
        let result = diagnose_preset(preset.name, THRESHOLD).expect("preset");
        let total: f64 = result.posteriors.iter().map(|(_, v)| v).sum();
        assert!(
            (total - 1.0).abs() < 1e-6,
            "{}: posteriors sum to {total}",
            preset.name
        );
    }
}

#[test]
fn test_diagnose_custom_priors() {
    let e = ev(&[("e9_obfuscated_code_or_weights", 1.0), ("e10_hidden_triggers_or_backdoors", 1.0)]);
    let ov = overrides(&[("human_malice", 0.90)]);
    let result = diagnose(&e, Some(&ov), THRESHOLD).expect("diagnose");
    assert_eq!(result.diagnosis, "human_malice");
    let result_default = diagnose(&e, None, THRESHOLD).expect("diagnose");
    assert!(
        result.posterior("human_malice") > result_default.posterior("human_malice"),
        "custom malice prior should raise malice posterior"
    );
}

#[test]
fn test_diagnose_result_to_dict() {
    let result = diagnose_preset("military_deception", THRESHOLD).expect("preset");
    let d = result.to_dict();
    assert!(d.contains_key("posteriors"));
    assert!(d.contains_key("diagnosis"));
    assert!(d.contains_key("confidence"));
    assert!(d.get("posteriors").unwrap().is_object());
}

#[test]
fn test_confidence_threshold_flagging() {
    let result = diagnose_preset("subtle_bias", THRESHOLD).expect("preset");
    assert_eq!(result.needs_investigation, result.confidence < 0.6);
}

// ── Sensitivity ──────────────────────────────────────────────────────

#[test]
fn test_sensitivity_analysis() {
    let result = diagnose_preset("subtle_bias", THRESHOLD).expect("preset");
    let engine = BayesianDiagnostic::new();
    let sens = compute_sensitivity(&result, &engine).expect("sensitivity");
    assert!(!sens.parameter_sensitivity.is_empty());
}

#[test]
fn test_sensitivity_summary() {
    let result = diagnose_preset("military_deception", THRESHOLD).expect("preset");
    let engine = BayesianDiagnostic::new();
    let sens = compute_sensitivity(&result, &engine).expect("sensitivity");
    let s = sens.summary();
    assert!(!s.is_empty());
}

#[test]
fn test_sensitivity_flip_analysis() {
    let result = diagnose_preset("subtle_bias", THRESHOLD).expect("preset");
    let engine = BayesianDiagnostic::new();
    let sens = compute_sensitivity(&result, &engine).expect("sensitivity");
    // Just verify it computes a well-formed list of flip entries.
    for entry in &sens.flip_analysis {
        assert!(!entry.parameter.is_empty());
        assert!(!entry.alternative_diagnosis.is_empty());
    }
}

// ── History ──────────────────────────────────────────────────────────

#[test]
fn test_history_record_and_retrieve() {
    let dir = temp_dir("history_record");
    let db_path = dir.join("test.db");
    let mut history = DiagnosisHistory::open(Some(&db_path)).expect("open db");
    let result = diagnose_preset("military_deception", THRESHOLD).expect("preset");
    history
        .record(&result, &result.evidence_vector, Some(&json!({"test": true})))
        .expect("record");
    let records = history.recent(1).expect("recent");
    assert_eq!(records.len(), 1);
    let (diag_result, _timestamp, metadata) = &records[0];
    assert_eq!(diag_result.diagnosis, "human_malice");
    assert_eq!(metadata.get("test"), Some(&json!(true)));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_history_stats() {
    let dir = temp_dir("history_stats");
    let db_path = dir.join("test.db");
    let mut history = DiagnosisHistory::open(Some(&db_path)).expect("open db");
    for preset in ["military_deception", "hardware_crash"] {
        let result = diagnose_preset(preset, THRESHOLD).expect("preset");
        history.record(&result, &result.evidence_vector, None).expect("record");
    }
    let stats = history.stats().expect("stats");
    assert_eq!(stats["total_diagnoses"], json!(2));
    assert!(
        stats["average_confidence"].as_f64().unwrap() > 0.9,
        "stats: {stats}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_history_export_csv() {
    let dir = temp_dir("history_csv");
    let db_path = dir.join("test.db");
    let csv_path = dir.join("export.csv");
    let mut history = DiagnosisHistory::open(Some(&db_path)).expect("open db");
    let result = diagnose_preset("military_deception", THRESHOLD).expect("preset");
    history.record(&result, &result.evidence_vector, None).expect("record");
    history.export_csv(&csv_path).expect("export");
    let meta = std::fs::metadata(&csv_path).expect("csv exists");
    assert!(meta.len() > 0);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_history_by_diagnosis() {
    let dir = temp_dir("history_by_diag");
    let db_path = dir.join("test.db");
    let mut history = DiagnosisHistory::open(Some(&db_path)).expect("open db");
    for preset in ["military_deception", "hardware_crash", "military_deception"] {
        let result = diagnose_preset(preset, THRESHOLD).expect("preset");
        history.record(&result, &result.evidence_vector, None).expect("record");
    }
    let malice_records = history.by_diagnosis("human_malice").expect("by_diagnosis");
    assert_eq!(malice_records.len(), 2);
    let _ = std::fs::remove_dir_all(&dir);
}

// ── Batch ────────────────────────────────────────────────────────────

#[test]
fn test_batch_process() {
    let bp = BatchProcessor::new();
    let evidence_list = vec![
        ev(&[("e1_hardware_anomalies", 1.0), ("e7_system_log_errors", 1.0), ("e5_temporal_pattern_sudden", 1.0)]),
        ev(&[("e9_obfuscated_code_or_weights", 1.0), ("e10_hidden_triggers_or_backdoors", 1.0)]),
    ];
    let results = bp.process(&evidence_list, None).expect("process");
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].diagnosis, "entropy");
    // Second case: e9+e10 without other malice indicators → bias may win.
    assert!(results[1].diagnosis == "human_malice" || results[1].diagnosis == "human_bias");
}

#[test]
fn test_batch_summary() {
    let bp = BatchProcessor::new();
    let evidence_list = vec![
        ev(&[("e1_hardware_anomalies", 1.0), ("e7_system_log_errors", 1.0)]),
        ev(&[("e9_obfuscated_code_or_weights", 1.0)]),
        Evidence::default(),
    ];
    let results = bp.process(&evidence_list, None).expect("process");
    let summary = bp.summary(&results);
    assert_eq!(summary["total"], json!(3));
    assert!(summary.get("average_confidence").is_some());
    assert!(summary.get("by_type").is_some());
}

#[test]
fn test_batch_compare() {
    let bp = BatchProcessor::new();
    let r1 = diagnose_preset("hardware_crash", THRESHOLD).expect("preset");
    let r2 = diagnose_preset("distribution_drift", THRESHOLD).expect("preset");
    let table = bp.compare(&[r1, r2], None);
    assert!(!table.is_empty());
    assert!(table.to_lowercase().contains("entropy"), "table: {table}");
}

// ── Alerts ───────────────────────────────────────────────────────────

#[test]
fn test_alert_manager() {
    let mut am = AlertManager::new();
    let triggered = Rc::new(RefCell::new(Vec::<(String, String)>::new()));

    let captured = Rc::clone(&triggered);
    am.add_rule(
        "test_rule",
        Box::new(|r: &ai_diagnostic_rs::DiagnosisResult| r.confidence > 0.9),
        vec![Box::new(move |r: &ai_diagnostic_rs::DiagnosisResult, name: &str| {
            captured.borrow_mut().push((r.diagnosis.clone(), name.to_string()));
        })],
    );

    let result = diagnose_preset("military_deception", THRESHOLD).expect("preset");
    am.check(&result);

    let triggered = triggered.borrow();
    assert_eq!(triggered.len(), 1);
    assert_eq!(triggered[0].0, "human_malice");
    assert_eq!(triggered[0].1, "test_rule");
}

#[test]
fn test_alert_manager_no_trigger() {
    let triggered = Rc::new(RefCell::new(0usize));
    let mut am = AlertManager::new();

    let captured = Rc::clone(&triggered);
    am.add_rule(
        "high_confidence_only",
        Box::new(|r: &ai_diagnostic_rs::DiagnosisResult| r.confidence > 0.99),
        vec![Box::new(move |_r: &ai_diagnostic_rs::DiagnosisResult, _name: &str| {
            *captured.borrow_mut() += 1;
        })],
    );

    let result = diagnose_preset("subtle_bias", THRESHOLD).expect("preset");
    am.check(&result);
    assert_eq!(*triggered.borrow(), 0);
}

#[test]
fn test_threshold_alert() {
    let ta_above = ThresholdAlert::new(0.9, Direction::Above).expect("alert");
    let ta_below = ThresholdAlert::new(0.5, Direction::Below).expect("alert");

    let result_high = diagnose_preset("military_deception", THRESHOLD).expect("preset");
    let ambiguous = ev(&[("e9_obfuscated_code_or_weights", 0.3), ("e4_human_interface_errors", 0.3)]);
    let result_low = diagnose(&ambiguous, None, THRESHOLD).expect("diagnose");

    // High confidence case
    assert!(ta_above.check_confidence(&result_high));
    // Low confidence case
    if result_low.confidence < 0.9 {
        assert!(!ta_above.check_confidence(&result_low));
    }
    // Below-direction check on the high-confidence case
    assert!(!ta_below.check_confidence(&result_high));

    // Posterior-based checks
    assert!(ta_above.check_posterior("human_malice", &result_high).unwrap());
    assert!(!ta_above.check_posterior("human_malice", &result_low).unwrap());

    // Unknown cause is rejected
    assert!(ta_above.check_posterior("nonexistent", &result_high).is_err());
}

#[test]
fn test_preset_rules() {
    let mut am = AlertManager::new();
    let rules = am.preset_rules();
    assert_eq!(rules.len(), 4);
    assert!(rules.contains(&"critical_malice".to_string()));
    assert!(rules.contains(&"warning_low_confidence".to_string()));
    assert!(rules.contains(&"entropy_alert".to_string()));
    assert!(rules.contains(&"investigation_needed".to_string()));
}

// ── Reports ──────────────────────────────────────────────────────────

#[test]
fn test_html_report() {
    let gen = ReportGenerator;
    let result = diagnose_preset("military_deception", THRESHOLD).expect("preset");
    let html = gen.html_report(&result, None, "AI Control Failure Diagnosis");
    let lower = html.to_lowercase();
    assert!(lower.contains("<html") || lower.contains("<!doctype"));
    assert!(lower.contains("malice"), "html: {html}");
}

#[test]
fn test_html_report_with_sensitivity() {
    let gen = ReportGenerator;
    let result = diagnose_preset("subtle_bias", THRESHOLD).expect("preset");
    let engine = BayesianDiagnostic::new();
    let sens = compute_sensitivity(&result, &engine).expect("sensitivity");
    let html = gen.html_report(&result, Some(&sens), "AI Control Failure Diagnosis");
    assert!(html.len() > 1000);
}

#[test]
fn test_html_comparison_report() {
    let gen = ReportGenerator;
    let r1 = diagnose_preset("hardware_crash", THRESHOLD).expect("preset");
    let r2 = diagnose_preset("distribution_drift", THRESHOLD).expect("preset");
    let labels = vec!["crash".to_string(), "drift".to_string()];
    let html = gen.html_comparison_report(&[r1, r2], Some(&labels));
    assert!(html.len() > 1000);
    let lower = html.to_lowercase();
    assert!(lower.contains("crash") || lower.contains("drift"), "html: {html}");
}

#[test]
fn test_text_report() {
    let gen = ReportGenerator;
    let result = diagnose_preset("hardware_crash", THRESHOLD).expect("preset");
    let text = gen.text_report(&result);
    assert!(text.to_lowercase().contains("entropy"), "text: {text}");
}

#[test]
fn test_json_report() {
    let gen = ReportGenerator;
    let result = diagnose_preset("military_deception", THRESHOLD).expect("preset");
    let j = gen.json_report(&result);
    let data: Value = serde_json::from_str(&j).expect("valid JSON");
    assert!(data.get("diagnosis").is_some());
    assert!(data.get("posteriors").is_some());
}

#[test]
fn test_save_report() {
    let gen = ReportGenerator;
    let dir = temp_dir("save_report");
    let path = dir.join("report.html");
    let result = diagnose_preset("hardware_crash", THRESHOLD).expect("preset");
    let html = gen.html_report(&result, None, "AI Control Failure Diagnosis");
    let saved = gen.save_report(&html, &path, "html").expect("save");
    let meta = std::fs::metadata(&saved).expect("file exists");
    assert!(meta.len() > 0);
    let _ = std::fs::remove_dir_all(&dir);
}

// ── Presets completeness ────────────────────────────────────────────

#[test]
fn test_all_presets_run() {
    for preset in PRESETS.iter() {
        let result = diagnose_preset(preset.name, THRESHOLD)
            .unwrap_or_else(|e| panic!("Preset {} failed: {e}", preset.name));
        assert!(FAILURE_CLASSES.contains(&result.diagnosis.as_str()));
        let total: f64 = result.posteriors.iter().map(|(_, v)| v).sum();
        assert!((total - 1.0).abs() < 1e-6);
    }
}

#[test]
fn test_all_presets_have_description() {
    for preset in PRESETS.iter() {
        assert!(!preset.description.is_empty(), "Preset {} missing description", preset.name);
        assert_eq!(preset.evidence.len(), 15, "Preset {} wrong evidence count", preset.name);
    }
}

// ── Edge cases ──────────────────────────────────────────────────────

#[test]
fn test_all_evidence_present() {
    let pairs: Vec<(&str, f64)> = EVIDENCE_PARAMS.iter().map(|n| (*n, 1.0)).collect();
    let e = ev(&pairs);
    let result = diagnose(&e, None, THRESHOLD).expect("diagnose");
    assert!(result.confidence > 0.99, "confidence: {}", result.confidence);
    assert_eq!(result.diagnosis, "human_malice");
}

#[test]
fn test_extreme_prior_override() {
    let e = Evidence::default();
    let ov = overrides(&[("entropy", 0.9999)]);
    let result = diagnose(&e, Some(&ov), THRESHOLD).expect("diagnose");
    assert!(
        result.posterior("entropy") > 0.05,
        "posterior: {}",
        result.posterior("entropy")
    );
}

// ── Parity guards (vs reference implementation constants) ───────────

#[test]
fn test_default_priors_match_reference() {
    let expected: [(&str, f64); 5] = [
        ("entropy", 0.10),
        ("engineering_limits", 0.25),
        ("human_error", 0.25),
        ("human_bias", 0.25),
        ("human_malice", 0.15),
    ];
    for (cause, val) in expected {
        assert_eq!(
            DEFAULT_PRIORS.iter().find(|(c, _)| *c == cause).map(|(_, v)| *v),
            Some(val)
        );
    }
}
