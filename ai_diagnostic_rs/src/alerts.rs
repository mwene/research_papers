//! Rule-based alerting with webhook, email, log-file, and console actions.

use crate::bayesian::DiagnosisResult;
use crate::likelihoods::{failure_class_label, FAILURE_CLASSES};
use serde_json::json;

#[derive(Debug, Clone)]
pub enum Direction {
    Above,
    Below,
}

/// Confidence/posterior threshold checks.
#[derive(Debug, Clone)]
pub struct ThresholdAlert {
    pub threshold: f64,
    pub direction: Direction,
}

impl ThresholdAlert {
    pub fn new(threshold: f64, direction: Direction) -> Result<Self, String> {
        if !(0.0..=1.0).contains(&threshold) {
            return Err(format!("threshold must be in [0, 1], got {threshold}"));
        }
        Ok(ThresholdAlert {
            threshold,
            direction,
        })
    }

    pub fn check_confidence(&self, result: &DiagnosisResult) -> bool {
        match self.direction {
            Direction::Above => result.confidence > self.threshold,
            Direction::Below => result.confidence < self.threshold,
        }
    }

    pub fn check_posterior(&self, cause: &str, result: &DiagnosisResult) -> Result<bool, String> {
        if !FAILURE_CLASSES.contains(&cause) {
            return Err(format!("Unknown cause: {cause}"));
        }
        let prob = result.posterior(cause);
        Ok(match self.direction {
            Direction::Above => prob > self.threshold,
            Direction::Below => prob < self.threshold,
        })
    }
}

pub type Condition = Box<dyn Fn(&DiagnosisResult) -> bool>;
pub type Action = Box<dyn Fn(&DiagnosisResult, &str)>;

pub struct AlertRule {
    pub name: String,
    pub condition: Condition,
    pub actions: Vec<Action>,
}

pub struct AlertManager {
    rules: Vec<AlertRule>,
    triggered: Vec<String>,
}

impl Default for AlertManager {
    fn default() -> Self {
        Self::new()
    }
}

impl AlertManager {
    pub fn new() -> Self {
        AlertManager {
            rules: Vec::new(),
            triggered: Vec::new(),
        }
    }

    pub fn add_rule(
        &mut self,
        name: impl Into<String>,
        condition: Condition,
        actions: Vec<Action>,
    ) {
        self.rules.push(AlertRule {
            name: name.into(),
            condition,
            actions,
        });
    }

    /// Check all rules against a result; fire actions of matching rules.
    /// Returns the names of triggered rules.
    pub fn check(&mut self, result: &DiagnosisResult) -> Vec<String> {
        self.triggered.clear();
        let rules = &mut self.rules;
        for rule in rules.iter_mut() {
            if (rule.condition)(result) {
                self.triggered.push(rule.name.clone());
                for action in &rule.actions {
                    action(result, &rule.name);
                }
            }
        }
        self.triggered.clone()
    }

    pub fn webhook_action(url: &str, method: &str) -> Action {
        let url = url.to_string();
        let method = method.to_string();
        Box::new(move |result: &DiagnosisResult, rule_name: &str| {
            let payload = json!({
                "rule": rule_name,
                "timestamp": chrono::Utc::now().to_rfc3339(),
                "posteriors": result
                    .posteriors
                    .iter()
                    .map(|(k, v)| (k.clone(), json!(v)))
                    .collect::<serde_json::Map<String, serde_json::Value>>(),
                "diagnosis": result.diagnosis,
                "diagnosis_label": failure_class_label(&result.diagnosis),
                "confidence": result.confidence,
                "needs_investigation": result.needs_investigation,
                "log_likelihoods": result
                    .log_likelihoods
                    .iter()
                    .map(|(k, v)| (k.clone(), json!(v)))
                    .collect::<serde_json::Map<String, serde_json::Value>>(),
            });
            match method.to_uppercase().as_str() {
                "POST" => ureq::post(&url)
                    .timeout(std::time::Duration::from_secs(10))
                    .send_json(payload),
                _ => ureq::request(&method.to_uppercase(), &url)
                    .timeout(std::time::Duration::from_secs(10))
                    .send_json(payload),
            }
            .map(|_| ())
            .unwrap_or_else(|e| {
                eprintln!("[AlertManager] webhook {rule_name:?} failed: {e}");
            });
        })
    }

    pub fn log_action(path: &str) -> Action {
        use std::io::Write;
        let path = path.to_string();
        Box::new(move |result: &DiagnosisResult, rule_name: &str| {
            let entry = json!({
                "timestamp": chrono::Utc::now().to_rfc3339(),
                "rule": rule_name,
                "posteriors": result
                    .posteriors
                    .iter()
                    .map(|(k, v)| (k.clone(), json!(v)))
                    .collect::<serde_json::Map<String, serde_json::Value>>(),
                "diagnosis": result.diagnosis,
                "diagnosis_label": failure_class_label(&result.diagnosis),
                "confidence": result.confidence,
                "needs_investigation": result.needs_investigation,
                "log_likelihoods": result
                    .log_likelihoods
                    .iter()
                    .map(|(k, v)| (k.clone(), json!(v)))
                    .collect::<serde_json::Map<String, serde_json::Value>>(),
            });
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
            {
                let _ = writeln!(f, "{entry}");
            }
        })
    }

    pub fn console_action() -> Action {
        Box::new(|result: &DiagnosisResult, rule_name: &str| {
            eprintln!(
                "[ALERT] {rule_name}: diagnosis={} confidence={:.2}%",
                result.diagnosis,
                result.confidence * 100.0
            );
        })
    }

    pub fn email_action(smtp_host: &str, to: &str, from_addr: &str) -> Action {
        use lettre::message::{Mailbox, Message};
        use lettre::{Address, SmtpTransport, Transport};

        let host = smtp_host.to_string();
        let to = to.to_string();
        let from_addr = from_addr.to_string();

        Box::new(move |result: &DiagnosisResult, rule_name: &str| {
            let run = || -> Result<(), String> {
                // smtp_host may include a port ("host:port"); default to 25 like smtplib.SMTP.
                let (host_part, port) = match host.rsplit_once(':') {
                    Some((h, p)) if p.parse::<u16>().is_ok() => {
                        (h.to_string(), p.parse::<u16>().unwrap())
                    }
                    _ => (host.clone(), 25),
                };

                let to_address: Address = to.parse().map_err(|_| format!("invalid To address: {to}"))?;
                let from_address: Address = from_addr
                    .parse()
                    .map_err(|_| format!("invalid From address: {from_addr}"))?;

                let mut body = format!(
                    "Alert triggered: {rule_name}\n\
                     Diagnosis: {}\n\
                     Confidence: {:.2}%\n\
                     Needs investigation: {}\n\n\
                     Posterior probabilities:\n",
                    result.diagnosis,
                    result.confidence * 100.0,
                    result.needs_investigation
                );
                let mut sorted = result.posteriors.clone();
                sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                for (cause, prob) in &sorted {
                    body.push_str(&format!("  {cause}: {prob:.4}\n"));
                }

                let email = Message::builder()
                    .from(Mailbox::new(None, from_address))
                    .to(Mailbox::new(None, to_address))
                    .subject(format!("[AI Diagnostic Alert] {rule_name}"))
                    .body(body)
                    .map_err(|e| e.to_string())?;

                let mailer = SmtpTransport::builder_dangerous(&host_part)
                    .port(port)
                    .timeout(Some(std::time::Duration::from_secs(10)))
                    .build();
                mailer.send(&email).map_err(|e| e.to_string())?;
                Ok(())
            };
            if let Err(e) = run() {
                eprintln!("[AlertManager] email to {to} failed: {e}");
            }
        })
    }

    /// The four built-in rules.
    pub fn preset_rules(&mut self) -> Vec<String> {
        let presets: Vec<(&str, Condition)> = vec![
            (
                "critical_malice",
                Box::new(|r: &DiagnosisResult| r.diagnosis == "human_malice" && r.confidence > 0.9),
            ),
            (
                "warning_low_confidence",
                Box::new(|r: &DiagnosisResult| r.confidence < 0.5),
            ),
            (
                "entropy_alert",
                Box::new(|r: &DiagnosisResult| r.diagnosis == "entropy" && r.confidence > 0.8),
            ),
            (
                "investigation_needed",
                Box::new(|r: &DiagnosisResult| r.needs_investigation),
            ),
        ];
        let mut names = Vec::new();
        for (name, condition) in presets {
            self.add_rule(name, condition, vec![Self::console_action()]);
            names.push(name.to_string());
        }
        names
    }
}
