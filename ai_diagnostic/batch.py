import csv
import json
from collections import Counter
from typing import Optional

from .evidence import Evidence, EVIDENCE_PARAMS, evidence_from_dict
from .bayesian import BayesianDiagnostic, DiagnosisResult
from .likelihoods import FAILURE_CLASSES, FAILURE_CLASS_LABELS


class BatchProcessor:
    def __init__(self, engine: Optional[BayesianDiagnostic] = None):
        self.engine = engine or BayesianDiagnostic()

    def process(self, evidence_list: list[Evidence], priors: dict = None) -> list[DiagnosisResult]:
        return [
            self.engine.diagnose(evidence, prior_overrides=priors)
            for evidence in evidence_list
        ]

    def process_from_dicts(self, dicts: list[dict]) -> list[DiagnosisResult]:
        return self.process([evidence_from_dict(d) for d in dicts])

    def summary(self, results: list[DiagnosisResult]) -> dict:
        if not results:
            return {
                "total": 0,
                "by_type": {},
                "average_confidence": 0.0,
                "flagged_for_investigation": 0,
                "max_confidence": 0.0,
                "min_confidence": 0.0,
            }

        type_counts = Counter(r.diagnosis for r in results)
        confidences = [r.confidence for r in results]
        flagged = sum(1 for r in results if r.needs_investigation)

        return {
            "total": len(results),
            "by_type": {
                FAILURE_CLASS_LABELS.get(k, k): v
                for k, v in type_counts.most_common()
            },
            "average_confidence": round(sum(confidences) / len(confidences), 4),
            "max_confidence": round(max(confidences), 4),
            "min_confidence": round(min(confidences), 4),
            "flagged_for_investigation": flagged,
        }

    def compare(self, results: list[DiagnosisResult]) -> str:
        if not results:
            return ""

        headers = ["Case", "Diagnosis", "Confidence", "Investigate?"]
        for cause in FAILURE_CLASSES:
            label = FAILURE_CLASS_LABELS.get(cause, cause)
            headers.append(label)

        rows = []
        for i, r in enumerate(results, 1):
            row = [
                str(i),
                FAILURE_CLASS_LABELS.get(r.diagnosis, r.diagnosis),
                f"{r.confidence:.2%}",
                "Yes" if r.needs_investigation else "No",
            ]
            for cause in FAILURE_CLASSES:
                row.append(f"{r.posteriors.get(cause, 0.0):.4f}")
            rows.append(row)

        col_widths = [max(len(h), max((len(row[j]) for row in rows), default=0))
                      for j, h in enumerate(headers)]

        def fmt_row(vals):
            return "  ".join(v.ljust(w) for v, w in zip(vals, col_widths))

        lines = [
            fmt_row(headers),
            "  ".join("-" * w for w in col_widths),
        ]
        for row in rows:
            lines.append(fmt_row(row))
        return "\n".join(lines)

    def export_batch(self, results: list[DiagnosisResult], path: str, format: str = "csv"):
        if format == "json":
            data = []
            for i, r in enumerate(results, 1):
                entry = {
                    "case": i,
                    "diagnosis": r.diagnosis,
                    "diagnosis_label": FAILURE_CLASS_LABELS.get(r.diagnosis, r.diagnosis),
                    "confidence": round(r.confidence, 4),
                    "needs_investigation": r.needs_investigation,
                    "posteriors": {k: round(v, 4) for k, v in r.posteriors.items()},
                }
                data.append(entry)
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        else:
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                header = ["case", "diagnosis", "diagnosis_label", "confidence", "needs_investigation"]
                header += [FAILURE_CLASS_LABELS.get(c, c) for c in FAILURE_CLASSES]
                writer.writerow(header)
                for i, r in enumerate(results, 1):
                    row = [
                        i,
                        r.diagnosis,
                        FAILURE_CLASS_LABELS.get(r.diagnosis, r.diagnosis),
                        f"{r.confidence:.4f}",
                        r.needs_investigation,
                    ]
                    row += [f"{r.posteriors.get(c, 0.0):.4f}" for c in FAILURE_CLASSES]
                    writer.writerow(row)
