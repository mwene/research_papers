//! HTML and text report generation for diagnostic results.

use crate::bayesian::DiagnosisResult;
use crate::evidence::{evidence_short_name, EVIDENCE_PARAMS};
use crate::likelihoods::{failure_class_label, FAILURE_CLASSES};
use crate::sensitivity::SensitivityReport;
use serde_json::{json, Map, Value};
use std::io::Write;

const SEVERITY_COLORS: [(&str, &str); 5] = [
    ("entropy", "#d97706"),
    ("engineering_limits", "#2563eb"),
    ("human_error", "#6b7280"),
    ("human_bias", "#8b5cf6"),
    ("human_malice", "#dc2626"),
];

fn severity_color(cause: &str) -> &'static str {
    SEVERITY_COLORS
        .iter()
        .find(|(c, _)| *c == cause)
        .map(|(_, v)| *v)
        .unwrap_or("#6b7280")
}

const CSS: &str = r#"
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  color: #1a1a1a; background: #f8fafc; line-height: 1.6; padding: 2rem;
}
.container { max-width: 960px; margin: 0 auto; background: #fff; border-radius: 8px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1); padding: 2rem; }
h1 { font-size: 1.5rem; margin-bottom: 0.25rem; }
.timestamp { color: #64748b; font-size: 0.875rem; margin-bottom: 1.5rem; }
h2 { font-size: 1.125rem; margin: 1.5rem 0 0.75rem; color: #334155; border-bottom: 1px solid #e2e8f0;
  padding-bottom: 0.375rem; }
table { width: 100%; border-collapse: collapse; font-size: 0.875rem; margin-bottom: 1rem; }
th { text-align: left; padding: 0.5rem 0.75rem; background: #f1f5f9; color: #475569;
  font-weight: 600; border-bottom: 2px solid #e2e8f0; }
td { padding: 0.5rem 0.75rem; border-bottom: 1px solid #f1f5f9; }
.bar-row { display: flex; align-items: center; margin-bottom: 0.375rem; }
.bar-label { width: 200px; font-size: 0.8125rem; color: #334155; flex-shrink: 0; text-align: right;
  padding-right: 0.75rem; }
.bar-track { flex: 1; height: 1.25rem; background: #f1f5f9; border-radius: 4px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 4px; transition: width 0.3s; }
.bar-value { width: 60px; text-align: right; font-size: 0.8125rem; font-weight: 600;
  padding-left: 0.5rem; }
.diagnosis-box { margin: 1rem 0; padding: 1rem 1.25rem; border-radius: 6px; border-left: 4px solid;
  background: #fefce8; }
.diagnosis-box.critical { background: #fef2f2; border-color: #dc2626; }
.diagnosis-box.warning { background: #fffbeb; border-color: #d97706; }
.diagnosis-box.info { background: #eff6ff; border-color: #2563eb; }
.diagnosis-box.ok { background: #f0fdf4; border-color: #16a34a; }
.severity-tag { display: inline-block; padding: 0.125rem 0.5rem; border-radius: 9999px;
  font-size: 0.75rem; font-weight: 600; color: #fff; margin-left: 0.5rem; }
ul.rec { list-style: none; padding: 0; }
ul.rec li { padding: 0.375rem 0; padding-left: 1.25rem; position: relative; font-size: 0.875rem; }
ul.rec li::before { content: "\2022"; position: absolute; left: 0; color: #2563eb; }
.meta { font-size: 0.8125rem; color: #64748b; }
@media print { body { padding: 0; } .container { box-shadow: none; } }
"#;

pub fn recommendations_for(diagnosis: &str) -> &'static [&'static str] {
    match diagnosis {
        "entropy" => &[
            "Run hardware diagnostics (memory, CPU, cooling).",
            "Check for cosmic-ray bit-flip reports or ECC errors.",
            "Review power supply stability logs.",
            "Consider physical inspection of affected components.",
        ],
        "engineering_limits" => &[
            "Review model architecture for known scaling limitations.",
            "Audit training data for distribution coverage gaps.",
            "Run adversarial robustness tests.",
            "Compare against known failure modes in similar architectures.",
        ],
        "human_error" => &[
            "Audit recent configuration changes and deployment logs.",
            "Review prompt templates and constraint definitions.",
            "Check for recent personnel changes or access modifications.",
            "Implement additional validation gates before deployment.",
        ],
        "human_bias" => &[
            "Audit training data for systematic biases.",
            "Review reward function design and objective specifications.",
            "Conduct fairness and bias testing across subgroups.",
            "Implement ongoing monitoring for bias drift.",
        ],
        "human_malice" => &[
            "IMMEDIATE: Isolate affected systems from network.",
            "Preserve all logs and artifacts for forensic analysis.",
            "Initiate security incident response protocol.",
            "Review access controls and code review processes.",
            "Engage legal counsel for contractual and regulatory obligations.",
        ],
        _ => &[],
    }
}

fn evidence_value_label(val: f64) -> String {
    if val == 0.0 {
        "absent".to_string()
    } else if val >= 0.9 {
        "present".to_string()
    } else {
        format!("{:.0}%", val * 100.0)
    }
}

fn diagnosis_class(result: &DiagnosisResult) -> &'static str {
    if result.confidence < 0.5 || result.needs_investigation {
        return "warning";
    }
    if result.diagnosis == "human_malice" && result.confidence > 0.9 {
        return "critical";
    }
    if result.diagnosis == "entropy" && result.confidence > 0.8 {
        return "info";
    }
    "ok"
}

pub struct ReportGenerator;

impl Default for ReportGenerator {
    fn default() -> Self {
        Self
    }
}

impl ReportGenerator {
    pub fn html_report(
        &self,
        result: &DiagnosisResult,
        sensitivity: Option<&SensitivityReport>,
        title: &str,
    ) -> String {
        let now = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC");
        let diag_label = failure_class_label(&result.diagnosis);
        let diag_color = severity_color(&result.diagnosis);
        let box_class = diagnosis_class(result);

        let mut evidence_rows = String::new();
        let ev = &result.evidence_vector;
        for name in EVIDENCE_PARAMS {
            let val = ev.get(name).unwrap();
            let short = evidence_short_name(name);
            let status = evidence_value_label(val);
            let present_cls = if val > 0.5 {
                " style=\"color:#dc2626;font-weight:600\""
            } else {
                ""
            };
            evidence_rows.push_str(&format!(
                "<tr><td>{short}</td><td>{val:.2}</td><td{present_cls}>{status}</td></tr>\n"
            ));
        }

        let mut sorted_posteriors: Vec<(&String, f64)> =
            result.posteriors.iter().map(|(c, v)| (c, *v)).collect();
        sorted_posteriors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut bar_rows = String::new();
        for (cause, prob) in &sorted_posteriors {
            let label = failure_class_label(cause);
            let color = severity_color(cause);
            let pct = prob * 100.0;
            bar_rows.push_str(&format!(
                "<div class=\"bar-row\"><span class=\"bar-label\">{label}</span>\
                 <span class=\"bar-track\"><span class=\"bar-fill\" \
                 style=\"width:{pct:.1}%;background:{color}\"></span></span>\
                 <span class=\"bar-value\">{:.1}%</span></div>\n",
                prob * 100.0
            ));
        }

        let mut sensitivity_section = String::new();
        if let Some(sens) = sensitivity {
            let mut rows = String::new();
            for (param, impact) in &sens.parameter_sensitivity {
                let short = evidence_short_name(param);
                rows.push_str(&format!("<tr><td>{short}</td><td>{impact:.4}</td></tr>\n"));
            }
            if !rows.is_empty() {
                sensitivity_section = format!(
                    "<h2>Sensitivity Analysis</h2>\n<table><thead><tr><th>Parameter</th>\
                     <th>Impact</th></tr></thead>\n<tbody>{rows}</tbody></table>\n"
                );
            }
        }

        let rec_items: String = recommendations_for(&result.diagnosis)
            .iter()
            .map(|r| format!("<li>{r}</li>\n"))
            .collect();

        format!(
            "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n\
             <meta charset=\"UTF-8\">\n\
             <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n\
             <title>{title}</title>\n<style>{CSS}</style>\n</head>\n<body>\n\
             <div class=\"container\">\n\
             <h1>{title}</h1>\n<p class=\"timestamp\">{now}</p>\n\
             <h2>Evidence Summary</h2>\n\
             <table><thead><tr><th>Parameter</th><th>Value</th><th>Status</th></tr></thead>\n\
             <tbody>{evidence_rows}</tbody></table>\n\
             <h2>Posterior Probabilities</h2>\n{bar_rows}\n\
             <div class=\"diagnosis-box {box_class}\">\n\
             <strong>Diagnosis:</strong> {diag_label} \
             <span class=\"severity-tag\" style=\"background:{diag_color}\">\
             {:.1}%</span>\n\
             <br><strong>Confidence:</strong> {:.2}%\n\
             <br><strong>Investigation required:</strong> {}\n\
             </div>\n{sensitivity_section}\
             <h2>Recommendations</h2>\n<ul class=\"rec\">{rec_items}</ul>\n\
             </div>\n</body>\n</html>",
            result.confidence * 100.0,
            result.confidence * 100.0,
            if result.needs_investigation { "Yes" } else { "No" },
        )
    }
}

impl ReportGenerator {
    pub fn html_comparison_report(
        &self,
        results: &[DiagnosisResult],
        labels: Option<&[String]>,
    ) -> String {
        let now = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC");
        let mut cards = String::new();

        for (i, result) in results.iter().enumerate() {
            let label = labels
                .and_then(|ls| ls.get(i))
                .map(|s| s.as_str())
                .unwrap_or("Unnamed");
            let diag_label = failure_class_label(&result.diagnosis);
            let diag_color = severity_color(&result.diagnosis);
            let box_class = diagnosis_class(result);

            let mut bar_rows = String::new();
            for cause in FAILURE_CLASSES {
                let prob = result.posterior(cause);
                let cl = failure_class_label(cause);
                let color = severity_color(cause);
                let pct = prob * 100.0;
                bar_rows.push_str(&format!(
                    "<div class=\"bar-row\"><span class=\"bar-label\">{cl}</span>\
                     <span class=\"bar-track\"><span class=\"bar-fill\" \
                     style=\"width:{pct:.1}%;background:{color}\"></span></span>\
                     <span class=\"bar-value\">{:.1}%</span></div>\n",
                    prob * 100.0
                ));
            }

            cards.push_str(&format!(
                "<div style=\"flex:1;min-width:300px;padding:1rem;border:1px solid #e2e8f0;\
                 border-radius:6px;margin:0.5rem\">\n<h3>{label}</h3>\n{bar_rows}\n\
                 <div class=\"diagnosis-box {box_class}\"><strong>{diag_label}</strong> \
                 <span class=\"severity-tag\" style=\"background:{diag_color}\">\
                 {:.1}%</span>\n</div>\n</div>\n",
                result.confidence * 100.0
            ));
        }

        format!(
            "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n\
             <meta charset=\"UTF-8\">\n\
             <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n\
             <title>Diagnosis Comparison</title>\n<style>{CSS}</style>\n</head>\n<body>\n\
             <div class=\"container\">\n<h1>Diagnosis Comparison</h1>\n\
             <p class=\"timestamp\">{now}</p>\n\
             <div style=\"display:flex;flex-wrap:wrap;gap:1rem\">\n{cards}\n</div>\n</div>\n</body>\n</html>"
        )
    }
}

impl ReportGenerator {
    pub fn text_report(&self, result: &DiagnosisResult) -> String {
        let now = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC");
        let mut lines = vec![
            "AI CONTROL FAILURE DIAGNOSIS".to_string(),
            format!("Generated: {now}"),
            String::new(),
            "Evidence:".to_string(),
        ];
        let ev = &result.evidence_vector;
        for name in EVIDENCE_PARAMS {
            let val = ev.get(name).unwrap();
            if val > 0.0 {
                lines.push(format!("  {}: {val:.2}", evidence_short_name(name)));
            }
        }
        if EVIDENCE_PARAMS.iter().all(|n| ev.get(n).unwrap() == 0.0) {
            lines.push("  (none)".to_string());
        }

        lines.push(String::new());
        lines.push("Posterior Probabilities:".to_string());
        let mut sorted: Vec<(&String, f64)> =
            result.posteriors.iter().map(|(c, v)| (c, *v)).collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (cause, prob) in &sorted {
            let label = failure_class_label(cause);
            let bar = "#".repeat((prob * 30.0) as usize);
            lines.push(format!("  {label:<35} {prob:>8.4}  {bar}"));
        }

        let diag_label = failure_class_label(&result.diagnosis);
        lines.push(String::new());
        lines.push(format!("DIAGNOSIS: {diag_label}"));
        lines.push(format!("CONFIDENCE: {:.2}%", result.confidence * 100.0));
        lines.push(format!(
            "Investigation required: {}",
            if result.needs_investigation { "Yes" } else { "No" }
        ));
        lines.push(String::new());
        lines.push("Recommendations:".to_string());
        for rec in recommendations_for(&result.diagnosis) {
            lines.push(format!("  - {rec}"));
        }

        lines.join("\n")
    }
}

impl ReportGenerator {
    pub fn json_report(&self, result: &DiagnosisResult) -> String {
        let mut data = Map::new();
        data.insert(
            "timestamp".into(),
            json!(chrono::Utc::now().to_rfc3339()),
        );
        data.insert("evidence".into(), Value::Object(result.evidence_vector.to_dict()));
        for (k, v) in result.to_dict() {
            data.insert(k, v);
        }
        data.insert(
            "recommendations".into(),
            json!(recommendations_for(&result.diagnosis)),
        );
        serde_json::to_string_pretty(&Value::Object(data)).unwrap_or_default()
    }

    pub fn save_report(
        &self,
        content: &str,
        path: &std::path::Path,
        format: &str,
    ) -> std::io::Result<std::path::PathBuf> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let ext = match format {
            "json" => ".json",
            "text" | "txt" => ".txt",
            _ => ".html",
        };
        let mut final_path = path.to_path_buf();
        if final_path.extension().is_none() {
            let mut s = final_path.into_os_string();
            s.push(ext);
            final_path = std::path::PathBuf::from(s);
        }
        let mut f = std::fs::File::create(&final_path)?;
        f.write_all(content.as_bytes())?;
        Ok(final_path)
    }
}
