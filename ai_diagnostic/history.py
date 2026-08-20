import csv
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from .evidence import Evidence, EVIDENCE_PARAMS
from .bayesian import BayesianDiagnostic, DiagnosisResult
from .likelihoods import FAILURE_CLASSES


class DiagnosisHistory:
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = str(Path.home() / ".ai_diagnostic" / "history.db")
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._ensure_table()

    def _ensure_table(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS diagnoses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                evidence_vector TEXT NOT NULL,
                posteriors TEXT NOT NULL,
                diagnosis TEXT NOT NULL,
                confidence REAL NOT NULL,
                metadata TEXT
            )
        """)
        self._conn.commit()

    def _row_to_result(self, row) -> tuple[DiagnosisResult, str, dict]:
        evidence_dict = json.loads(row["evidence_vector"])
        evidence = Evidence(**{k: evidence_dict.get(k, 0.0) for k in EVIDENCE_PARAMS})
        posteriors = json.loads(row["posteriors"])
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}

        result = DiagnosisResult(
            posteriors=posteriors,
            evidence_vector=evidence,
            diagnosis=row["diagnosis"],
            confidence=row["confidence"],
            needs_investigation=row["confidence"] < 0.6,
            log_likelihoods={},
        )
        return result, row["timestamp"], metadata

    def record(self, result: DiagnosisResult, evidence: Evidence, metadata: dict = None):
        self._conn.execute(
            "INSERT INTO diagnoses (timestamp, evidence_vector, posteriors, diagnosis, confidence, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                datetime.now().isoformat(),
                json.dumps(evidence.to_dict()),
                json.dumps(result.posteriors),
                result.diagnosis,
                result.confidence,
                json.dumps(metadata) if metadata else None,
            ),
        )
        self._conn.commit()

    def recent(self, n: int = 10) -> list[tuple[DiagnosisResult, str, dict]]:
        rows = self._conn.execute(
            "SELECT * FROM diagnoses ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        return [self._row_to_result(row) for row in rows]

    def by_diagnosis(self, diagnosis_type: str) -> list[tuple[DiagnosisResult, str, dict]]:
        rows = self._conn.execute(
            "SELECT * FROM diagnoses WHERE diagnosis = ? ORDER BY id DESC",
            (diagnosis_type,),
        ).fetchall()
        return [self._row_to_result(row) for row in rows]

    def by_timerange(self, start: datetime, end: datetime) -> list[tuple[DiagnosisResult, str, dict]]:
        rows = self._conn.execute(
            "SELECT * FROM diagnoses WHERE timestamp >= ? AND timestamp <= ? ORDER BY id DESC",
            (start.isoformat(), end.isoformat()),
        ).fetchall()
        return [self._row_to_result(row) for row in rows]

    def stats(self) -> dict:
        row = self._conn.execute(
            "SELECT COUNT(*) as total, AVG(confidence) as avg_confidence "
            "FROM diagnoses"
        ).fetchone()
        total = row["total"]
        avg_confidence = row["avg_confidence"] or 0.0

        type_rows = self._conn.execute(
            "SELECT diagnosis, COUNT(*) as cnt, AVG(confidence) as avg_conf "
            "FROM diagnoses GROUP BY diagnosis ORDER BY cnt DESC"
        ).fetchall()

        flagged = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM diagnoses WHERE confidence < 0.6"
        ).fetchone()["cnt"]

        return {
            "total_diagnoses": total,
            "average_confidence": round(avg_confidence, 4),
            "flagged_for_investigation": flagged,
            "by_type": [
                {
                    "diagnosis": r["diagnosis"],
                    "count": r["cnt"],
                    "average_confidence": round(r["avg_conf"], 4),
                }
                for r in type_rows
            ],
        }

    def export_csv(self, path: str):
        rows = self._conn.execute("SELECT * FROM diagnoses ORDER BY id").fetchall()
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "timestamp", "diagnosis", "confidence", "metadata", "evidence_vector", "posteriors"])
            for row in rows:
                writer.writerow([
                    row["id"],
                    row["timestamp"],
                    row["diagnosis"],
                    row["confidence"],
                    row["metadata"] or "",
                    row["evidence_vector"],
                    row["posteriors"],
                ])

    def search(self, query: str) -> list[tuple[DiagnosisResult, str, dict]]:
        rows = self._conn.execute(
            "SELECT * FROM diagnoses WHERE metadata LIKE ? ORDER BY id DESC",
            (f"%{query}%",),
        ).fetchall()
        return [self._row_to_result(row) for row in rows]

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
