//! Diagnostic interface: preset scenarios, high-level API, interactive CLI.

use crate::bayesian::{BayesianDiagnostic, DiagnosisResult};
use crate::evidence::{evidence_short_name, Evidence};
use crate::likelihoods::{failure_class_label, MILITARY_PRIORS};
use std::collections::HashMap;
use std::io::{self, BufRead, Write};

pub struct ScenarioPreset {
    pub name: &'static str,
    pub description: &'static str,
    pub evidence: [(&'static str, f64); 15],
    pub priors: Option<&'static [(&'static str, f64)]>,
}

pub static PRESETS: [ScenarioPreset; 5] = [
    ScenarioPreset {
        name: "military_deception",
        description: "Military AI generating deceptive outputs favoring one faction, despite being ostensibly neutral. Code obfuscated, hidden triggers found, data exfiltration detected, designer has intelligence ties.",
        evidence: [
            ("e1_hardware_anomalies", 0.0),
            ("e2_data_distribution_shift", 0.0),
            ("e3_model_architecture_irregularities", 0.0),
            ("e4_human_interface_errors", 0.0),
            ("e5_temporal_pattern_sudden", 1.0),
            ("e6_output_coherence_adversarial", 1.0),
            ("e7_system_log_errors", 0.0),
            ("e8_external_environment_changes", 0.0),
            ("e9_obfuscated_code_or_weights", 1.0),
            ("e10_hidden_triggers_or_backdoors", 1.0),
            ("e11_unexplained_communication_channels", 1.0),
            ("e12_behavioral_context_inconsistency", 1.0),
            ("e13_unusual_training_data", 1.0),
            ("e14_designer_history_red_flags", 1.0),
            ("e15_legal_or_contractual_violations", 1.0),
        ],
        priors: Some(&MILITARY_PRIORS),
    },
    ScenarioPreset {
        name: "hardware_crash",
        description: "Sudden system crash with hardware temperature spikes and bit-flip reports in logs.",
        evidence: [
            ("e1_hardware_anomalies", 1.0),
            ("e2_data_distribution_shift", 0.0),
            ("e3_model_architecture_irregularities", 0.0),
            ("e4_human_interface_errors", 0.0),
            ("e5_temporal_pattern_sudden", 1.0),
            ("e6_output_coherence_adversarial", 0.0),
            ("e7_system_log_errors", 1.0),
            ("e8_external_environment_changes", 0.0),
            ("e9_obfuscated_code_or_weights", 0.0),
            ("e10_hidden_triggers_or_backdoors", 0.0),
            ("e11_unexplained_communication_channels", 0.0),
            ("e12_behavioral_context_inconsistency", 0.0),
            ("e13_unusual_training_data", 0.0),
            ("e14_designer_history_red_flags", 0.0),
            ("e15_legal_or_contractual_violations", 0.0),
        ],
        priors: None,
    },
    ScenarioPreset {
        name: "distribution_drift",
        description: "AI producing coherent but misaligned outputs after encountering out-of-distribution inputs. No hardware issues, no code anomalies.",
        evidence: [
            ("e1_hardware_anomalies", 0.0),
            ("e2_data_distribution_shift", 1.0),
            ("e3_model_architecture_irregularities", 1.0),
            ("e4_human_interface_errors", 0.0),
            ("e5_temporal_pattern_sudden", 0.0),
            ("e6_output_coherence_adversarial", 1.0),
            ("e7_system_log_errors", 0.5),
            ("e8_external_environment_changes", 0.0),
            ("e9_obfuscated_code_or_weights", 0.0),
            ("e10_hidden_triggers_or_backdoors", 0.0),
            ("e11_unexplained_communication_channels", 0.0),
            ("e12_behavioral_context_inconsistency", 0.0),
            ("e13_unusual_training_data", 0.0),
            ("e14_designer_history_red_flags", 0.0),
            ("e15_legal_or_contractual_violations", 0.0),
        ],
        priors: None,
    },
    ScenarioPreset {
        name: "misconfigured_reward",
        description: "AI slowly drifting from intended behavior due to reward function misconfiguration. Human error suspected.",
        evidence: [
            ("e1_hardware_anomalies", 0.0),
            ("e2_data_distribution_shift", 0.3),
            ("e3_model_architecture_irregularities", 0.3),
            ("e4_human_interface_errors", 1.0),
            ("e5_temporal_pattern_sudden", 0.0),
            ("e6_output_coherence_adversarial", 0.2),
            ("e7_system_log_errors", 0.3),
            ("e8_external_environment_changes", 0.5),
            ("e9_obfuscated_code_or_weights", 0.0),
            ("e10_hidden_triggers_or_backdoors", 0.0),
            ("e11_unexplained_communication_channels", 0.0),
            ("e12_behavioral_context_inconsistency", 0.0),
            ("e13_unusual_training_data", 0.0),
            ("e14_designer_history_red_flags", 0.0),
            ("e15_legal_or_contractual_violations", 0.0),
        ],
        priors: None,
    },
    ScenarioPreset {
        name: "subtle_bias",
        description: "AI producing systematically biased outputs across many queries. No single dramatic failure, but consistent pattern of misalignment.",
        evidence: [
            ("e1_hardware_anomalies", 0.0),
            ("e2_data_distribution_shift", 0.3),
            ("e3_model_architecture_irregularities", 0.0),
            ("e4_human_interface_errors", 0.0),
            ("e5_temporal_pattern_sudden", 0.0),
            ("e6_output_coherence_adversarial", 0.8),
            ("e7_system_log_errors", 0.0),
            ("e8_external_environment_changes", 0.3),
            ("e9_obfuscated_code_or_weights", 0.0),
            ("e10_hidden_triggers_or_backdoors", 0.0),
            ("e11_unexplained_communication_channels", 0.0),
            ("e12_behavioral_context_inconsistency", 0.0),
            ("e13_unusual_training_data", 0.0),
            ("e14_designer_history_red_flags", 0.0),
            ("e15_legal_or_contractual_violations", 0.0),
        ],
        priors: None,
    },
];

pub fn find_preset(name: &str) -> Option<&'static ScenarioPreset> {
    PRESETS.iter().find(|p| p.name == name)
}

fn evidence_from_pairs(pairs: &[(&'static str, f64)]) -> Result<Evidence, String> {
    let mut ev = Evidence::default();
    for (name, val) in pairs {
        ev.set(name, *val)?;
    }
    Ok(ev)
}

impl ScenarioPreset {
    pub fn evidence(&self) -> Result<Evidence, String> {
        evidence_from_pairs(&self.evidence)
    }

    pub fn priors_map(&self) -> Option<HashMap<String, f64>> {
        self.priors.map(|rows| {
            rows.iter()
                .map(|(c, v)| (c.to_string(), *v))
                .collect::<HashMap<String, f64>>()
        })
    }
}

/// Run a diagnostic on an evidence vector with a fresh default engine.
pub fn diagnose(
    evidence: &Evidence,
    prior_overrides: Option<&HashMap<String, f64>>,
    confidence_threshold: f64,
) -> Result<DiagnosisResult, String> {
    BayesianDiagnostic::new().diagnose(evidence, prior_overrides, confidence_threshold)
}

/// Run a diagnostic on a named preset scenario.
pub fn diagnose_preset(preset_name: &str, confidence_threshold: f64) -> Result<DiagnosisResult, String> {
    let preset = find_preset(preset_name).ok_or_else(|| {
        format!(
            "Unknown preset: {preset_name}. Available: {}",
            PRESETS.iter().map(|p| p.name).collect::<Vec<_>>().join(", ")
        )
    })?;
    let evidence = preset.evidence()?;
    diagnose(&evidence, preset.priors_map().as_ref(), confidence_threshold)
}

/// Interactive diagnostic session (mirrors the Python CLI flow).
pub fn diagnose_interactive() -> Result<DiagnosisResult, String> {
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    let _ = writeln!(stdout, "{}", "=".repeat(60));
    let _ = writeln!(stdout, "  AI CONTROL FAILURE DIAGNOSTIC TOOL");
    let _ = writeln!(stdout, "  Based on: Human versus AI — The Gödelian Referee");
    let _ = writeln!(stdout, "{}", "=".repeat(60));
    let _ = writeln!(stdout);
    let _ = writeln!(stdout, "Available presets:");
    for preset in PRESETS.iter() {
        let short_desc: String = preset.description.chars().take(50).collect();
        let _ = writeln!(stdout, "  {:<30} {}...", preset.name, short_desc);
    }
    let _ = writeln!(stdout);
    let _ = writeln!(stdout, "Or enter custom evidence values.");
    let _ = writeln!(stdout);

    let _ = write!(stdout, "Enter preset name (or 'custom'): ");
    let _ = stdout.flush();
    let mut choice = String::new();
    let _ = stdin.lock().read_line(&mut choice);
    let choice = choice.trim().to_lowercase();

    if find_preset(&choice).is_some() {
        let result = diagnose_preset(&choice, 0.6)?;
        let _ = writeln!(stdout);
        let _ = writeln!(stdout, "{result}");
        return Ok(result);
    }

    if choice != "custom" {
        let _ = writeln!(stdout, "Unknown option: {choice}. Proceeding with custom input.");
    }

    let _ = writeln!(stdout);
    let _ = writeln!(stdout, "Enter evidence values for each parameter.");
    let _ = writeln!(stdout, "  0 = evidence absent, 1 = evidence fully present");
    let _ = writeln!(stdout, "  Press Enter to skip (defaults to 0)");
    let _ = writeln!(stdout);

    let mut evidence = Evidence::default();
    for name in crate::evidence::EVIDENCE_PARAMS {
        let _ = write!(stdout, "  {}: ", evidence_short_name(name));
        let _ = stdout.flush();
        let mut line = String::new();
        let _ = stdin.lock().read_line(&mut line);
        let val_str = line.trim();
        if val_str.is_empty() {
            continue;
        }
        match val_str.parse::<f64>() {
            Ok(v) => {
                if let Err(e) = evidence.set(name, v) {
                    let _ = writeln!(stdout, "    {e}, defaulting to 0.0");
                }
            }
            Err(_) => {
                let _ = writeln!(stdout, "    Invalid value '{val_str}', defaulting to 0.0");
            }
        }
    }

    let _ = writeln!(stdout);
    let _ = write!(stdout, "Use custom priors? (y/N): ");
    let _ = stdout.flush();
    let mut priors_line = String::new();
    let _ = stdin.lock().read_line(&mut priors_line);

    let mut priors: HashMap<String, f64> = HashMap::new();
    if priors_line.trim().eq_ignore_ascii_case("y") {
        use crate::likelihoods::{FAILURE_CLASSES, DEFAULT_PRIORS};
        for cause in FAILURE_CLASSES {
            let label = failure_class_label(cause);
            let _ = write!(stdout, "  P({label}): ");
            let _ = stdout.flush();
            let mut line = String::new();
            let _ = stdin.lock().read_line(&mut line);
            if let Ok(v) = line.trim().parse::<f64>() {
                priors.insert(cause.to_string(), v);
            } else {
                // fall back to the table's current value
                if let Some(default_val) = DEFAULT_PRIORS
                    .iter()
                    .find(|(c, _)| *c == cause)
                    .map(|(_, v)| *v)
                {
                    priors.insert(cause.to_string(), default_val);
                }
            }
        }
    }

    let result = diagnose(&evidence, Some(&priors), 0.6)?;
    let _ = writeln!(stdout);
    let _ = writeln!(stdout, "{result}");
    Ok(result)
}
