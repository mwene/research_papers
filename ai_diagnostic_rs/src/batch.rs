//! Batch processing, summaries, side-by-side comparison tables, exports.

use crate::bayesian::{BayesianDiagnostic, DiagnosisResult};
use crate::evidence::{Evidence, evidence_from_map};
use crate::likelihoods::{failure_class_label, FAILURE_CLASSES};
use serde_json::{json, Map, Value};
use std::collections::HashMap;

pub struct BatchProcessor {
    pub engine: BayesianDiagnostic,
}

impl Default for BatchProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl BatchProcessor {
    pub fn new() -> Self {
        BatchProcessor {
            engine: BayesianDiagnostic::new(),
        }
    }

    pub fn with_engine(engine: BayesianDiagnostic) -> Self {
        BatchProcessor { engine }
    }

    pub fn process(
        &self,
        evidence_list: &[Evidence],
        priors: Option<&HashMap<String, f64>>,
    ) -> Result<Vec<DiagnosisResult>, String> {
        evidence_list
            .iter()
            .map(|e| self.engine.diagnose(e, priors, 0.6))
            .collect()
    }

    pub fn process_from_dicts(
        &self,
        dicts: &[Map<String, Value>],
    ) -> Result<Vec<DiagnosisResult>, String> {
        let mut all = Vec::with_capacity(dicts.len());
        for d in dicts {
            let ev = evidence_from_map(d)?;
            all.push(self.engine.diagnose(&ev, None, 0.6)?);
        }
        Ok(all)
    }

    pub fn summary(&self, results: &[DiagnosisResult]) -> Value {
        if results.is_empty() {
            return json!({
                "total": 0,
                "by_type": {},
                "average_confidence": 0.0,
                "max_confidence": 0.0,
                "min_confidence": 0.0,
                "flagged_for_investigation": 0,
            });
        }

        let mut type_counts: Vec<(String, usize)> = Vec::new();
        for r in results {
            match type_counts.iter_mut().find(|(c, _)| c == &r.diagnosis) {
                Some((_, n)) => *n += 1,
                None => type_counts.push((r.diagnosis.clone(), 1)),
            }
        }
        type_counts.sort_by_key(|(_, n)| std::cmp::Reverse(*n));

        let confidences: Vec<f64> = results.iter().map(|r| r.confidence).collect();
        let avg = confidences.iter().sum::<f64>() / confidences.len() as f64;
        let round4 = |x: f64| (x * 10000.0).round() / 10000.0;

        json!({
            "total": results.len(),
            "by_type": type_counts
                .into_iter()
                .map(|(c, n)| (failure_class_label(&c).to_string(), json!(n)))
                .collect::<Map<String, Value>>(),
            "average_confidence": round4(avg),
            "max_confidence": round4(confidences.iter().cloned().fold(f64::MIN, f64::max)),
            "min_confidence": round4(confidences.iter().cloned().fold(f64::MAX, f64::min)),
            "flagged_for_investigation": results.iter().filter(|r| r.needs_investigation).count(),
        })
    }

    /// Side-by-side comparison table. `labels` optionally names each case.
    pub fn compare(
        &self,
        results: &[DiagnosisResult],
        labels: Option<&[String]>,
    ) -> String {
        if results.is_empty() {
            return String::new();
        }

        let mut headers: Vec<String> = vec![
            "Case".to_string(),
            "Diagnosis".to_string(),
            "Confidence".to_string(),
            "Investigate?".to_string(),
        ];
        for cause in FAILURE_CLASSES {
            headers.push(failure_class_label(cause).to_string());
        }

        let mut rows: Vec<Vec<String>> = Vec::new();
        for (i, r) in results.iter().enumerate() {
            let label = labels
                .and_then(|ls| ls.get(i))
                .cloned()
                .unwrap_or_else(|| i.to_string());
            let mut row = vec![
                label,
                failure_class_label(&r.diagnosis).to_string(),
                format!("{:.2}%", r.confidence * 100.0),
                if r.needs_investigation { "Yes" } else { "No" }.to_string(),
            ];
            for cause in FAILURE_CLASSES {
                row.push(format!("{:.4}", r.posterior(cause)));
            }
            rows.push(row);
        }

        let col_widths: Vec<usize> = headers
            .iter()
            .enumerate()
            .map(|(j, h)| {
                rows.iter()
                    .map(|row| row[j].chars().count())
                    .chain(std::iter::once(h.chars().count()))
                    .max()
                    .unwrap_or(0)
            })
            .collect();

        let fmt_row = |vals: &[String]| -> String {
            vals.iter()
                .zip(col_widths.iter())
                .map(|(v, w)| {
                    let pad = w.saturating_sub(v.chars().count());
                    format!("{v}{}", " ".repeat(pad))
                })
                .collect::<Vec<_>>()
                .join("  ")
        };

        let mut lines = vec![fmt_row(&headers)];
        lines.push(col_widths
            .iter()
            .map(|w| "-".repeat(*w))
            .collect::<Vec<_>>()
            .join("  "));
        for row in &rows {
            lines.push(fmt_row(row));
        }
        lines.join("\n")
    }

    pub fn export_batch(
        &self,
        results: &[DiagnosisResult],
        path: &std::path::Path,
        format: &str,
    ) -> Result<(), String> {
        match format {
            "json" => {
                let data: Vec<Value> = results
                    .iter()
                    .enumerate()
                    .map(|(i, r)| {
                        let round4 =
                            |x: f64| (x * 10000.0).round() / 10000.0;
                        json!({
                            "case": i + 1,
                            "diagnosis": r.diagnosis,
                            "diagnosis_label": failure_class_label(&r.diagnosis),
                            "confidence": round4(r.confidence),
                            "needs_investigation": r.needs_investigation,
                            "posteriors": r
                                .posteriors
                                .iter()
                                .map(|(k, v)| (k.clone(), json!(round4(*v))))
                                .collect::<Map<String, Value>>(),
                        })
                    })
                    .collect();
                serde_json::to_writer_pretty(std::fs::File::create(path).map_err(|e| e.to_string())?, &data)
                    .map_err(|e| e.to_string())
            }
            _ => {
                let mut w = csv::Writer::create(path).map_err(|e| e.to_string())?;
                let mut header: Vec<String> = vec![
                    "case".into(),
                    "diagnosis".into(),
                    "diagnosis_label".into(),
                    "confidence".into(),
                    "needs_investigation".into(),
                ];
                header.extend(FAILURE_CLASSES.map(|c| failure_class_label(c).to_string()));
                w.row(&header);
                for (i, r) in results.iter().enumerate() {
                    let mut row = vec![
                        (i + 1).to_string(),
                        r.diagnosis.clone(),
                        failure_class_label(&r.diagnosis).to_string(),
                        format!("{:.4}", r.confidence),
                        r.needs_investigation.to_string(),
                    ];
                    row.extend(FAILURE_CLASSES.map(|c| format!("{:.4}", r.posterior(c))));
                    w.row(&row);
                }
                w.flush().map_err(|e| e.to_string())
            }
        }
    }
}

mod csv {
    use std::fs::File;
    use std::io::{BufWriter, Write};
    use std::path::Path;

    pub struct Writer {
        inner: BufWriter<File>,
    }

    impl Writer {
        pub fn create(path: &Path) -> std::io::Result<Self> {
            Ok(Writer {
                inner: BufWriter::new(File::create(path)?),
            })
        }

        pub fn row(&mut self, fields: &[String]) {
            let line = fields
                .iter()
                .map(|f| {
                    if f.contains(',') || f.contains('"') || f.contains('\n') {
                        format!("\"{}\"", f.replace('"', "\"\""))
                    } else {
                        f.clone()
                    }
                })
                .collect::<Vec<_>>()
                .join(",");
            let _ = writeln!(self.inner, "{line}");
        }

        pub fn flush(mut self) -> std::io::Result<()> {
            self.inner.flush()
        }
    }
}
