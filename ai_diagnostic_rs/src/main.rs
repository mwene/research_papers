//! AI Diagnostic Tool — comprehensive CLI (port of `__main__.py`).
//!
//! Usage:
//!     ai_diagnostic diagnose --preset military_deception
//!     ai_diagnostic diagnose --interactive
//!     ai_diagnostic batch --input cases.json --output results.json
//!     ai_diagnostic sensitivity --preset military_deception
//!     ai_diagnostic compare --preset1 hardware_crash --preset2 distribution_drift
//!     ai_diagnostic report --preset military_deception --format html --output report.html
//!     ai_diagnostic history --recent 10
//!     ai_diagnostic serve --port 8000
//!     ai_diagnostic presets
//!     ai_diagnostic params

use ai_diagnostic_rs::{
    diagnose, diagnose_interactive, diagnose_preset, BatchProcessor, BayesianDiagnostic,
    DiagnosisHistory, Evidence, ReportGenerator, EVIDENCE_PARAMS,
};
use clap::{CommandFactory, Parser, Subcommand};
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "ai_diagnostic",
    version = ai_diagnostic_rs::VERSION,
    about = "AI Control Failure Diagnostic Framework"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Run a diagnosis
    Diagnose {
        #[arg(long)]
        preset: Option<String>,
        /// JSON file with evidence parameters
        #[arg(long = "evidence-file")]
        evidence_file: Option<PathBuf>,
        #[arg(long)]
        interactive: bool,
        /// Include sensitivity analysis
        #[arg(long, short = 's')]
        sensitivity: bool,
        /// Record to history
        #[arg(long)]
        history: bool,
    },
    /// Run sensitivity analysis
    Sensitivity {
        #[arg(long)]
        preset: String,
    },
    /// Compare two scenarios side by side
    Compare {
        #[arg(long = "preset1")]
        preset1: String,
        #[arg(long = "preset2")]
        preset2: String,
    },
    /// Batch process cases from a JSON file
    Batch {
        #[arg(short, long)]
        input: PathBuf,
        #[arg(short, long)]
        output: Option<PathBuf>,
        #[arg(short, long, default_value = "json")]
        format: String,
    },
    /// Generate a report (html/json/text)
    Report {
        #[arg(long)]
        preset: String,
        #[arg(long, default_value = "html")]
        format: String,
        #[arg(short, long)]
        output: Option<String>,
    },
    /// View diagnosis history
    History {
        /// Show last N records
        #[arg(long, short = 'n')]
        recent: Option<i64>,
        /// Filter by diagnosis type
        #[arg(long = "type-filter")]
        type_filter: Option<String>,
        /// Show statistics
        #[arg(long)]
        stats: bool,
        /// Export to CSV
        #[arg(long = "export-csv")]
        export_csv: Option<PathBuf>,
    },
    /// List available presets
    Presets,
    /// List evidence parameters
    Params,
    /// Start the REST API server
    Serve {
        #[arg(long, default_value = "0.0.0.0")]
        host: String,
        #[arg(long, default_value_t = 8000)]
        port: u16,
    },
}

fn load_evidence_file(path: &PathBuf) -> Result<(Evidence, Option<HashMap<String, f64>>), String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let data: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let obj = data.as_object().ok_or("Evidence file must contain a JSON object")?;

    // Accept either {"evidence": {...}, "priors": {...}} or a flat evidence object.
    let evidence_obj = obj.get("evidence").and_then(Value::as_object).unwrap_or(obj);
    let evidence: Evidence =
        serde_json::from_value(Value::Object(evidence_obj.clone())).map_err(|e| e.to_string())?;
    let priors = obj.get("priors").and_then(Value::as_object).map(|p| {
        p.iter()
            .filter_map(|(k, v)| v.as_f64().map(|f| (k.clone(), f)))
            .collect::<HashMap<String, f64>>()
    });
    Ok((evidence, priors))
}

fn cmd_diagnose(args: &DiagnoseArgs) -> Result<(), String> {
    if args.interactive {
        return diagnose_interactive().map(|_| ());
    }

    let result = if let Some(preset) = &args.preset {
        diagnose_preset(preset, 0.6)?
    } else if let Some(file) = &args.evidence_file {
        let (evidence, priors) = load_evidence_file(file)?;
        diagnose(&evidence, priors.as_ref(), 0.6)?
    } else {
        return Err(
            "Error: specify --preset, --evidence-file, or --interactive".to_string(),
        );
    };

    print!("{result}");

    if args.sensitivity {
        let engine = BayesianDiagnostic::new();
        let sens = ai_diagnostic_rs::compute_sensitivity(&result, &engine)?;
        println!();
        println!("{}", sens.summary());
    }

    if args.history {
        let mut history = DiagnosisHistory::open(None).map_err(|e| e.to_string())?;
        let id = history
            .record(&result, &result.evidence_vector, None)
            .map_err(|e| e.to_string())?;
        println!("\nRecorded to history (id={id})");
    }
    Ok(())
}

struct DiagnoseArgs {
    preset: Option<String>,
    evidence_file: Option<PathBuf>,
    interactive: bool,
    sensitivity: bool,
    history: bool,
}

fn main() {
    let cli = Cli::parse();
    let exit_code = match run(cli) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("{e}");
            1
        }
    };
    std::process::exit(exit_code);
}

fn run(cli: Cli) -> Result<(), String> {
    match cli.command {
        None => {
            Cli::command().print_help().map_err(|e| e.to_string())?;
            println!();
            Err("Error: no command given".to_string())
        }
        Some(Command::Diagnose {
            preset,
            evidence_file,
            interactive,
            sensitivity,
            history,
        }) => cmd_diagnose(&DiagnoseArgs {
            preset,
            evidence_file,
            interactive,
            sensitivity,
            history,
        }),
        Some(Command::Sensitivity { preset }) => {
            let result = diagnose_preset(&preset, 0.6)?;
            let engine = BayesianDiagnostic::new();
            let sens = ai_diagnostic_rs::compute_sensitivity(&result, &engine)?;
            println!("{}", sens.summary());
            Ok(())
        }
        Some(Command::Compare { preset1, preset2 }) => {
            let r1 = diagnose_preset(&preset1, 0.6)?;
            let r2 = diagnose_preset(&preset2, 0.6)?;
            let bp = BatchProcessor::new();
            println!(
                "{}",
                bp.compare(&[r1, r2], Some(&[preset1.clone(), preset2.clone()]))
            );
            Ok(())
        }
        Some(Command::Batch {
            input,
            output,
            format,
        }) => {
            let text = std::fs::read_to_string(&input).map_err(|e| e.to_string())?;
            let data: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
            let cases: Vec<Value> = match &data {
                Value::Array(items) => items.clone(),
                Value::Object(obj) => obj
                    .get("cases")
                    .or_else(|| obj.get("evidence"))
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default(),
                _ => Vec::new(),
            };

            let mut evidence_list = Vec::with_capacity(cases.len());
            for case in &cases {
                let ev: Evidence = serde_json::from_value(case.clone())
                    .map_err(|e| format!("Invalid case: {e}"))?;
                evidence_list.push(ev);
            }

            let bp = BatchProcessor::new();
            let results = bp.process(&evidence_list, None)?;
            let summary = bp.summary(&results);

            println!("Processed {} cases", summary["total"].as_u64().unwrap_or(0));
            println!(
                "Diagnoses: {}",
                summary["by_type"]
                    .as_object()
                    .map(|m| m
                        .iter()
                        .map(|(k, v)| format!("{k}: {}", v))
                        .collect::<Vec<_>>()
                        .join(", "))
                    .unwrap_or_default()
            );
            println!(
                "Average confidence: {:.2}%",
                summary["average_confidence"].as_f64().unwrap_or(0.0) * 100.0
            );
            println!(
                "Flagged for investigation: {}",
                summary["flagged_for_investigation"]
                    .as_u64()
                    .unwrap_or(0)
            );

            if let Some(out_path) = output {
                bp.export_batch(&results, &out_path, &format)
                    .map_err(|e| e.to_string())?;
                println!("Exported to {}", out_path.display());
            }
            Ok(())
        }
        Some(Command::Report {
            preset,
            format,
            output,
        }) => {
            let result = diagnose_preset(&preset, 0.6)?;
            let gen = ReportGenerator;

            let content;
            let fmt_for_save;
            match format.as_str() {
                "html" => {
                    let engine = BayesianDiagnostic::new();
                    let sens = ai_diagnostic_rs::compute_sensitivity(&result, &engine)?;
                    content = gen.html_report(&result, Some(&sens), "AI Control Failure Diagnosis");
                    fmt_for_save = "html";
                }
                "json" => {
                    content = gen.json_report(&result);
                    fmt_for_save = "json";
                }
                _ => {
                    content = gen.text_report(&result);
                    fmt_for_save = "text";
                }
            }

            match output {
                Some(path) => {
                    let saved = gen
                        .save_report(&content, std::path::Path::new(&path), fmt_for_save)
                        .map_err(|e| e.to_string())?;
                    println!("Report saved to {}", saved.display());
                }
                None => println!("{content}"),
            }
            Ok(())
        }
        Some(Command::History {
            recent,
            type_filter,
            stats,
            export_csv,
        }) => {
            let history = DiagnosisHistory::open(None).map_err(|e| e.to_string())?;

            if stats {
                let stats = history.stats().map_err(|e| e.to_string())?;
                println!(
                    "{}",
                    serde_json::to_string_pretty(&stats).map_err(|e| e.to_string())?
                );
                return Ok(());
            }

            let records = if let Some(n) = recent {
                history.recent(n).map_err(|e| e.to_string())?
            } else if let Some(ref t) = type_filter {
                history.by_diagnosis(t).map_err(|e| e.to_string())?
            } else {
                history.recent(20).map_err(|e| e.to_string())?
            };

            if records.is_empty() {
                println!("No records found.");
                return Ok(());
            }

            println!(
                "{:<6} {:<20} {:<25} {:<12}",
                "ID", "Timestamp", "Diagnosis", "Confidence"
            );
            println!("{}", "-".repeat(65));
            for (result, timestamp, _) in &records {
                let ts: String = timestamp.chars().take(19).collect();
                println!(
                    "{:<6} {:<20} {:<25} {:<12.2}",
                    "",
                    ts,
                    result.diagnosis,
                    result.confidence * 100.0
                );
            }

            if let Some(csv_path) = export_csv {
                history.export_csv(&csv_path).map_err(|e| e.to_string())?;
                println!("Exported to {}", csv_path.display());
            }
            Ok(())
        }
        Some(Command::Presets) => {
            println!("Available presets:\n");
            for preset in ai_diagnostic_rs::PRESETS.iter() {
                println!("  {}", preset.name);
                println!("    {}", preset.description);
                let active = preset.evidence.iter().filter(|(_, v)| *v > 0.0).count();
                println!("    Evidence parameters active: {active}/15");
                println!();
            }
            Ok(())
        }
        Some(Command::Params) => {
            println!("Evidence Parameters:\n");
            for name in EVIDENCE_PARAMS {
                println!("  {name}");
                println!("    Short: {}", ai_diagnostic_rs::evidence_short_name(name));
                println!("    Desc:  {}", ai_diagnostic_rs::evidence_description(name));
                println!();
            }
            Ok(())
        }
        Some(Command::Serve { host, port }) => {
            let rt = tokio::runtime::Runtime::new().map_err(|e| e.to_string())?;
            rt.block_on(async move {
                ai_diagnostic_rs::api::serve(&host, port)
                    .await
                    .map_err(|e| e.to_string())
            })
        }
    }
}
