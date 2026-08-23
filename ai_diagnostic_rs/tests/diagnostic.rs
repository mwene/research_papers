//! Port of `test_diagnostic.py`: quick validation against the paper's
//! example scenario (Section 9: Malicious AI Detection — military context).

use ai_diagnostic_rs::{diagnose, diagnose_preset, Evidence};

const THRESHOLD: f64 = 0.6;

#[test]
fn test_military_deception_preset() {
    // The paper's worked example: military AI with all malice indicators.
    let result = diagnose_preset("military_deception", THRESHOLD).expect("preset should run");

    println!("=== Paper Example: Military Deception ===\n{result}\n");

    assert_eq!(
        result.diagnosis, "human_malice",
        "Expected human_malice, got {}",
        result.diagnosis
    );
    assert!(
        result.confidence > 0.99,
        "Expected >99% confidence, got {:.2}%",
        result.confidence * 100.0
    );
}

#[test]
fn test_hardware_crash() {
    // Should diagnose as entropy.
    let result = diagnose_preset("hardware_crash", THRESHOLD).expect("preset should run");

    println!("=== Hardware Crash Scenario ===\n{result}\n");

    assert_eq!(
        result.diagnosis, "entropy",
        "Expected entropy, got {}",
        result.diagnosis
    );
}

#[test]
fn test_distribution_drift() {
    // Should diagnose as engineering limits.
    let result = diagnose_preset("distribution_drift", THRESHOLD).expect("preset should run");

    println!("=== Distribution Drift Scenario ===\n{result}\n");

    assert_eq!(
        result.diagnosis, "engineering_limits",
        "Expected engineering_limits, got {}",
        result.diagnosis
    );
}

#[test]
fn test_custom_evidence() {
    // Test with a custom evidence vector.
    let evidence = Evidence {
        e9_obfuscated_code_or_weights: 0.8,
        e10_hidden_triggers_or_backdoors: 0.7,
        e14_designer_history_red_flags: 0.9,
        ..Default::default()
    };
    let result = diagnose(&evidence, None, THRESHOLD).expect("diagnose should run");

    println!("=== Custom Evidence (partial malice indicators) ===\n{result}\n");
    println!("Top diagnosis: {} ({:.2}%)", result.diagnosis, result.confidence * 100.0);
}

#[test]
fn test_empty_evidence() {
    // Test with no evidence — should use priors only.
    let evidence = Evidence::default();
    let result = diagnose(&evidence, None, THRESHOLD).expect("diagnose should run");

    println!("=== Empty Evidence (priors only) ===\n{result}\n");
}
