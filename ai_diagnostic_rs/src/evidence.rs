//! Evidence vector definitions: 15 observable parameters, each in [0, 1].

use serde::de::Deserializer;
use serde::Serialize;
use serde_json::{Map, Value};
use std::fmt;

pub const EVIDENCE_PARAMS: [&str; 15] = [
    "e1_hardware_anomalies",
    "e2_data_distribution_shift",
    "e3_model_architecture_irregularities",
    "e4_human_interface_errors",
    "e5_temporal_pattern_sudden",
    "e6_output_coherence_adversarial",
    "e7_system_log_errors",
    "e8_external_environment_changes",
    "e9_obfuscated_code_or_weights",
    "e10_hidden_triggers_or_backdoors",
    "e11_unexplained_communication_channels",
    "e12_behavioral_context_inconsistency",
    "e13_unusual_training_data",
    "e14_designer_history_red_flags",
    "e15_legal_or_contractual_violations",
];

pub fn evidence_description(name: &str) -> &'static str {
    match name {
        "e1_hardware_anomalies" => {
            "Hardware anomalies (temperature spikes, bit-flip reports, power fluctuations)"
        }
        "e2_data_distribution_shift" => {
            "Training/data distribution shift (out-of-distribution inputs)"
        }
        "e3_model_architecture_irregularities" => {
            "Reward/model architecture irregularities (gradient explosion, loss instability)"
        }
        "e4_human_interface_errors" => {
            "Human interface errors (misconfigured parameters, wrong prompts, incorrect constraints)"
        }
        "e5_temporal_pattern_sudden" => "Temporal pattern (sudden vs gradual onset)",
        "e6_output_coherence_adversarial" => {
            "Output coherence (internally consistent vs gibberish vs adversarial)"
        }
        "e7_system_log_errors" => "System logs (error codes, warnings, permission violations)",
        "e8_external_environment_changes" => {
            "External environment changes (new data sources, connectivity changes)"
        }
        "e9_obfuscated_code_or_weights" => {
            "Obfuscated code or weights (deliberate complexity to hide function)"
        }
        "e10_hidden_triggers_or_backdoors" => {
            "Hidden triggers or backdoors (inputs that activate unintended behavior)"
        }
        "e11_unexplained_communication_channels" => {
            "Unexplained communication channels (exfiltration of data, covert signaling)"
        }
        "e12_behavioral_context_inconsistency" => {
            "Behavioral inconsistency under different contexts (aligned in testing, malicious in deployment)"
        }
        "e13_unusual_training_data" => {
            "Unusual training data (data that teaches deception or harmful behaviors)"
        }
        "e14_designer_history_red_flags" => {
            "Designer history or affiliation (known malicious actors, state actors, covert operations)"
        }
        "e15_legal_or_contractual_violations" => {
            "Legal or contractual violations (terms of service breaches, export control violations)"
        }
        _ => "",
    }
}

pub fn evidence_short_name(name: &str) -> &'static str {
    match name {
        "e1_hardware_anomalies" => "Hardware anomalies",
        "e2_data_distribution_shift" => "Data distribution shift",
        "e3_model_architecture_irregularities" => "Model architecture irregularities",
        "e4_human_interface_errors" => "Human interface errors",
        "e5_temporal_pattern_sudden" => "Sudden temporal pattern",
        "e6_output_coherence_adversarial" => "Adversarial output coherence",
        "e7_system_log_errors" => "System log errors",
        "e8_external_environment_changes" => "External environment changes",
        "e9_obfuscated_code_or_weights" => "Obfuscated code/weights",
        "e10_hidden_triggers_or_backdoors" => "Hidden triggers/backdoors",
        "e11_unexplained_communication_channels" => "Unexplained communication channels",
        "e12_behavioral_context_inconsistency" => "Behavioral context inconsistency",
        "e13_unusual_training_data" => "Unusual training data",
        "e14_designer_history_red_flags" => "Designer history red flags",
        "e15_legal_or_contractual_violations" => "Legal/contractual violations",
        _ => "Unknown parameter",
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Default)]
pub struct Evidence {
    #[serde(default)]
    pub e1_hardware_anomalies: f64,
    #[serde(default)]
    pub e2_data_distribution_shift: f64,
    #[serde(default)]
    pub e3_model_architecture_irregularities: f64,
    #[serde(default)]
    pub e4_human_interface_errors: f64,
    #[serde(default)]
    pub e5_temporal_pattern_sudden: f64,
    #[serde(default)]
    pub e6_output_coherence_adversarial: f64,
    #[serde(default)]
    pub e7_system_log_errors: f64,
    #[serde(default)]
    pub e8_external_environment_changes: f64,
    #[serde(default)]
    pub e9_obfuscated_code_or_weights: f64,
    #[serde(default)]
    pub e10_hidden_triggers_or_backdoors: f64,
    #[serde(default)]
    pub e11_unexplained_communication_channels: f64,
    #[serde(default)]
    pub e12_behavioral_context_inconsistency: f64,
    #[serde(default)]
    pub e13_unusual_training_data: f64,
    #[serde(default)]
    pub e14_designer_history_red_flags: f64,
    #[serde(default)]
    pub e15_legal_or_contractual_violations: f64,
}

impl Evidence {
    pub fn validate(&self) -> Result<(), String> {
        for name in EVIDENCE_PARAMS {
            let val = self.get(name).unwrap();
            if !(0.0..=1.0).contains(&val) {
                return Err(format!("{name} must be in [0, 1], got {val}"));
            }
        }
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<f64> {
        match name {
            "e1_hardware_anomalies" => Some(self.e1_hardware_anomalies),
            "e2_data_distribution_shift" => Some(self.e2_data_distribution_shift),
            "e3_model_architecture_irregularities" => Some(self.e3_model_architecture_irregularities),
            "e4_human_interface_errors" => Some(self.e4_human_interface_errors),
            "e5_temporal_pattern_sudden" => Some(self.e5_temporal_pattern_sudden),
            "e6_output_coherence_adversarial" => Some(self.e6_output_coherence_adversarial),
            "e7_system_log_errors" => Some(self.e7_system_log_errors),
            "e8_external_environment_changes" => Some(self.e8_external_environment_changes),
            "e9_obfuscated_code_or_weights" => Some(self.e9_obfuscated_code_or_weights),
            "e10_hidden_triggers_or_backdoors" => Some(self.e10_hidden_triggers_or_backdoors),
            "e11_unexplained_communication_channels" => Some(self.e11_unexplained_communication_channels),
            "e12_behavioral_context_inconsistency" => Some(self.e12_behavioral_context_inconsistency),
            "e13_unusual_training_data" => Some(self.e13_unusual_training_data),
            "e14_designer_history_red_flags" => Some(self.e14_designer_history_red_flags),
            "e15_legal_or_contractual_violations" => Some(self.e15_legal_or_contractual_violations),
            _ => None,
        }
    }

    pub fn set(&mut self, name: &str, value: f64) -> Result<(), String> {
        if !(0.0..=1.0).contains(&value) {
            return Err(format!("{name} must be in [0, 1], got {value}"));
        }
        match name {
            "e1_hardware_anomalies" => self.e1_hardware_anomalies = value,
            "e2_data_distribution_shift" => self.e2_data_distribution_shift = value,
            "e3_model_architecture_irregularities" => self.e3_model_architecture_irregularities = value,
            "e4_human_interface_errors" => self.e4_human_interface_errors = value,
            "e5_temporal_pattern_sudden" => self.e5_temporal_pattern_sudden = value,
            "e6_output_coherence_adversarial" => self.e6_output_coherence_adversarial = value,
            "e7_system_log_errors" => self.e7_system_log_errors = value,
            "e8_external_environment_changes" => self.e8_external_environment_changes = value,
            "e9_obfuscated_code_or_weights" => self.e9_obfuscated_code_or_weights = value,
            "e10_hidden_triggers_or_backdoors" => self.e10_hidden_triggers_or_backdoors = value,
            "e11_unexplained_communication_channels" => self.e11_unexplained_communication_channels = value,
            "e12_behavioral_context_inconsistency" => self.e12_behavioral_context_inconsistency = value,
            "e13_unusual_training_data" => self.e13_unusual_training_data = value,
            "e14_designer_history_red_flags" => self.e14_designer_history_red_flags = value,
            "e15_legal_or_contractual_violations" => self.e15_legal_or_contractual_violations = value,
            _ => return Err(format!("Unknown evidence parameter: {name}")),
        }
        Ok(())
    }

    pub fn with(mut self, name: &str, value: f64) -> Result<Self, String> {
        self.set(name, value)?;
        Ok(self)
    }

    pub fn to_vector(&self) -> Vec<f64> {
        EVIDENCE_PARAMS.iter().map(|n| self.get(n).unwrap()).collect()
    }

    pub fn to_dict(&self) -> Map<String, Value> {
        let mut map = Map::new();
        for name in EVIDENCE_PARAMS {
            map.insert(
                name.to_string(),
                serde_json::json!(self.get(name).unwrap()),
            );
        }
        map
    }

    pub fn summary(&self) -> String {
        let mut lines = Vec::new();
        for name in EVIDENCE_PARAMS {
            let val = self.get(name).unwrap();
            if val > 0.0 {
                lines.push(format!("  {}: {val:.2}", evidence_short_name(name)));
            }
        }
        if lines.is_empty() {
            return "  (no evidence present)".to_string();
        }
        lines.join("\n")
    }
}

impl fmt::Display for Evidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

pub fn evidence_from_map(d: &Map<String, Value>) -> Result<Evidence, String> {
    let mut ev = Evidence::default();
    for name in EVIDENCE_PARAMS {
        if let Some(v) = d.get(name) {
            let val = v.as_f64().ok_or_else(|| {
                format!("{name} must be a number, got {v}")
            })?;
            ev.set(name, val)?;
        }
    }
    Ok(ev)
}

impl<'de> serde::Deserialize<'de> for Evidence {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = Map::<String, Value>::deserialize(deserializer)?;
        evidence_from_map(&raw).map_err(serde::de::Error::custom)
    }
}
