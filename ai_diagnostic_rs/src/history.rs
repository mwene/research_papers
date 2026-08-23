//! SQLite-backed diagnosis history.

use crate::bayesian::DiagnosisResult;
use crate::evidence::{Evidence, EVIDENCE_PARAMS};
use chrono::Local;
use rusqlite::{params, Connection, Row};
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};

pub type HistoryRecord = (DiagnosisResult, String, Value);

pub struct DiagnosisHistory {
    conn: Connection,
}

fn default_db_path() -> PathBuf {
    if let Ok(path) = std::env::var("HISTORY_DB_PATH") {
        return PathBuf::from(path);
    }
    let home = std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."));
    home.join(".ai_diagnostic").join("history.db")
}

fn row_to_record(row: &Row<'_>) -> rusqlite::Result<HistoryRecord> {
    let evidence_json: String = row.get("evidence_vector")?;
    let posteriors_json: String = row.get("posteriors")?;
    let diagnosis: String = row.get("diagnosis")?;
    let confidence: f64 = row.get("confidence")?;
    let timestamp: String = row.get("timestamp")?;
    let metadata_raw: Option<String> = row.get("metadata")?;

    let evidence_map: Map<String, Value> =
        serde_json::from_str(&evidence_json).unwrap_or_default();
    let mut ev = Evidence::default();
    for name in EVIDENCE_PARAMS {
        if let Some(v) = evidence_map.get(name).and_then(Value::as_f64) {
            let _ = ev.set(name, v);
        }
    }
    let posteriors: Vec<(String, f64)> =
        serde_json::from_str::<Map<String, Value>>(&posteriors_json)
            .map(|m| {
                m.into_iter()
                    .filter_map(|(k, v)| v.as_f64().map(|f| (k, f)))
                    .collect()
            })
            .unwrap_or_default();

    let result = DiagnosisResult {
        posteriors,
        evidence_vector: ev,
        diagnosis,
        confidence,
        needs_investigation: confidence < 0.6,
        log_likelihoods: Vec::new(),
    };
    let metadata: Value = metadata_raw
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or(Value::Object(Map::new()));
    Ok((result, timestamp, metadata))
}

impl DiagnosisHistory {
    pub fn open(db_path: Option<&Path>) -> rusqlite::Result<Self> {
        let path = db_path
            .map(Path::to_path_buf)
            .unwrap_or_else(default_db_path);
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let conn = Connection::open(path)?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS diagnoses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                evidence_vector TEXT NOT NULL,
                posteriors TEXT NOT NULL,
                diagnosis TEXT NOT NULL,
                confidence REAL NOT NULL,
                metadata TEXT
            )",
            [],
        )?;
        Ok(DiagnosisHistory { conn })
    }

    pub fn record(
        &mut self,
        result: &DiagnosisResult,
        evidence: &Evidence,
        metadata: Option<&Value>,
    ) -> rusqlite::Result<i64> {
        self.conn.execute(
            "INSERT INTO diagnoses (timestamp, evidence_vector, posteriors, diagnosis, confidence, metadata) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                Local::now().format("%Y-%m-%dT%H:%M:%S%.6f").to_string(),
                json!(evidence.to_dict()).to_string(),
                json!(result
                    .posteriors
                    .iter()
                    .map(|(k, v)| (k.clone(), json!(v)))
                    .collect::<Map<String, Value>>())
                .to_string(),
                result.diagnosis,
                result.confidence,
                metadata.map(Value::to_string),
            ],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    fn query_rows(&self, sql: &str, args: &[&dyn rusqlite::ToSql]) -> rusqlite::Result<Vec<HistoryRecord>> {
        let mut stmt = self.conn.prepare(sql)?;
        let rows = stmt.query_map(args, row_to_record)?;
        rows.collect()
    }

    pub fn recent(&self, n: i64) -> rusqlite::Result<Vec<HistoryRecord>> {
        self.query_rows(
            "SELECT * FROM diagnoses ORDER BY id DESC LIMIT ?1",
            &[&n],
        )
    }

    pub fn by_diagnosis(&self, diagnosis_type: &str) -> rusqlite::Result<Vec<HistoryRecord>> {
        self.query_rows(
            "SELECT * FROM diagnoses WHERE diagnosis = ?1 ORDER BY id DESC",
            &[&diagnosis_type],
        )
    }

    pub fn by_timerange(
        &self,
        start: &str,
        end: &str,
    ) -> rusqlite::Result<Vec<HistoryRecord>> {
        self.query_rows(
            "SELECT * FROM diagnoses WHERE timestamp >= ?1 AND timestamp <= ?2 ORDER BY id DESC",
            &[&start, &end],
        )
    }

    pub fn search(&self, query: &str) -> rusqlite::Result<Vec<HistoryRecord>> {
        let like = format!("%{query}%");
        self.query_rows(
            "SELECT * FROM diagnoses WHERE metadata LIKE ?1 ORDER BY id DESC",
            &[&like],
        )
    }

    pub fn stats(&self) -> rusqlite::Result<Value> {
        let agg = self.conn.query_row(
            "SELECT COUNT(*) as total, AVG(confidence) as avg_confidence FROM diagnoses",
            [],
            |row| {
                Ok((
                    row.get::<_, i64>("total")?,
                    row.get::<_, Option<f64>>("avg_confidence")?.unwrap_or(0.0),
                ))
            },
        )?;
        let (total, avg_confidence) = agg;

        let mut stmt = self.conn.prepare(
            "SELECT diagnosis, COUNT(*) as cnt, AVG(confidence) as avg_conf \
             FROM diagnoses GROUP BY diagnosis ORDER BY cnt DESC",
        )?;
        let type_rows = stmt
            .query_map([], |row| {
                Ok(json!({
                    "diagnosis": row.get::<_, String>("diagnosis")?,
                    "count": row.get::<_, i64>("cnt")?,
                    "average_confidence": (row.get::<_, f64>("avg_conf")? * 10000.0).round() / 10000.0,
                }))
            })?
            .collect::<rusqlite::Result<Vec<Value>>>()?;

        let flagged: i64 = self.conn.query_row(
            "SELECT COUNT(*) as cnt FROM diagnoses WHERE confidence < 0.6",
            [],
            |row| row.get("cnt"),
        )?;

        Ok(json!({
            "total_diagnoses": total,
            "average_confidence": (avg_confidence * 10000.0).round() / 10000.0,
            "flagged_for_investigation": flagged,
            "by_type": type_rows,
        }))
    }

    pub fn export_csv(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let mut stmt = self.conn.prepare("SELECT * FROM diagnoses ORDER BY id")?;
        let mut rows = stmt.query([])?;
        let mut w = csv_writer::Writer::create(path)?;
        w.row(&[
            "id".to_string(),
            "timestamp".to_string(),
            "diagnosis".to_string(),
            "confidence".to_string(),
            "metadata".to_string(),
            "evidence_vector".to_string(),
            "posteriors".to_string(),
        ]);
        while let Some(row) = rows.next()? {
            let id: i64 = row.get("id")?;
            let timestamp: String = row.get("timestamp")?;
            let diagnosis: String = row.get("diagnosis")?;
            let confidence: f64 = row.get("confidence")?;
            let metadata: Option<String> = row.get("metadata")?;
            let evidence_vector: String = row.get("evidence_vector")?;
            let posteriors: String = row.get("posteriors")?;
            w.row(&[
                id.to_string(),
                timestamp,
                diagnosis,
                format!("{confidence}"),
                metadata.unwrap_or_default(),
                evidence_vector,
                posteriors,
            ]);
        }
        Ok(w.flush()?)
    }

    pub fn close(self) {}
}

mod csv_writer {
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
