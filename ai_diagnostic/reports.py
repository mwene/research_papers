"""
HTML and text report generation for diagnostic results.

Produces styled, self-contained HTML reports with CSS bar charts,
plain-text summaries, and structured JSON output.
"""

import json
import os
from datetime import datetime, timezone
from typing import List, Optional

from .evidence import Evidence, EVIDENCE_PARAMS, EVIDENCE_SHORT_NAMES
from .bayesian import BayesianDiagnostic, DiagnosisResult
from .likelihoods import FAILURE_CLASSES, FAILURE_CLASS_LABELS

_SEVERITY_COLORS = {
    "entropy":            "#d97706",
    "engineering_limits": "#2563eb",
    "human_error":        "#6b7280",
    "human_bias":         "#8b5cf6",
    "human_malice":       "#dc2626",
}

_CSS = """\
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
ul.rec li::before { content: "\\2022"; position: absolute; left: 0; color: #2563eb; }
.meta { font-size: 0.8125rem; color: #64748b; }
@media print { body { padding: 0; } .container { box-shadow: none; } }
"""

_RECOMMENDATIONS = {
    "entropy": [
        "Run hardware diagnostics (memory, CPU, cooling).",
        "Check for cosmic-ray bit-flip reports or ECC errors.",
        "Review power supply stability logs.",
        "Consider physical inspection of affected components.",
    ],
    "engineering_limits": [
        "Review model architecture for known scaling limitations.",
        "Audit training data for distribution coverage gaps.",
        "Run adversarial robustness tests.",
        "Compare against known failure modes in similar architectures.",
    ],
    "human_error": [
        "Audit recent configuration changes and deployment logs.",
        "Review prompt templates and constraint definitions.",
        "Check for recent personnel changes or access modifications.",
        "Implement additional validation gates before deployment.",
    ],
    "human_bias": [
        "Audit training data for systematic biases.",
        "Review reward function design and objective specifications.",
        "Conduct fairness and bias testing across subgroups.",
        "Implement ongoing monitoring for bias drift.",
    ],
    "human_malice": [
        "IMMEDIATE: Isolate affected systems from network.",
        "Preserve all logs and artifacts for forensic analysis.",
        "Initiate security incident response protocol.",
        "Review access controls and code review processes.",
        "Engage legal counsel for contractual and regulatory obligations.",
    ],
}


def _evidence_value_label(val: float) -> str:
    if val == 0.0:
        return "absent"
    if val >= 0.9:
        return "present"
    return f"{val:.0%}"


def _diagnosis_class(result: DiagnosisResult) -> str:
    if result.confidence < 0.5:
        return "warning"
    if result.needs_investigation:
        return "warning"
    if result.diagnosis == "human_malice" and result.confidence > 0.9:
        return "critical"
    if result.diagnosis == "entropy" and result.confidence > 0.8:
        return "info"
    return "ok"


class ReportGenerator:
    """Generates styled HTML, plain-text, and JSON reports."""

    def html_report(
        self,
        result: DiagnosisResult,
        sensitivity=None,
        title: str = "AI Control Failure Diagnosis",
    ) -> str:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        diag_label = FAILURE_CLASS_LABELS.get(result.diagnosis, result.diagnosis)
        diag_color = _SEVERITY_COLORS.get(result.diagnosis, "#6b7280")
        box_class = _diagnosis_class(result)

        evidence_rows = ""
        ev = result.evidence_vector
        for name in EVIDENCE_PARAMS:
            val = getattr(ev, name)
            short = EVIDENCE_SHORT_NAMES[name]
            status = _evidence_value_label(val)
            present_cls = ' style="color:#dc2626;font-weight:600"' if val > 0.5 else ""
            evidence_rows += (
                f'<tr><td>{short}</td>'
                f'<td>{val:.2f}</td>'
                f'<td{present_cls}>{status}</td></tr>\n'
            )

        sorted_posteriors = sorted(
            result.posteriors.items(), key=lambda x: x[1], reverse=True
        )
        bar_rows = ""
        for cause, prob in sorted_posteriors:
            label = FAILURE_CLASS_LABELS.get(cause, cause)
            color = _SEVERITY_COLORS.get(cause, "#6b7280")
            pct = prob * 100
            bar_rows += (
                f'<div class="bar-row">'
                f'<span class="bar-label">{label}</span>'
                f'<span class="bar-track">'
                f'<span class="bar-fill" style="width:{pct:.1f}%;background:{color}"></span>'
                f'</span>'
                f'<span class="bar-value">{prob:.1%}</span>'
                f'</div>\n'
            )

        sensitivity_section = ""
        if sensitivity is not None:
            rows = ""
            if isinstance(sensitivity, dict):
                for param, impact in sensitivity.items():
                    short = EVIDENCE_SHORT_NAMES.get(param, param)
                    rows += f"<tr><td>{short}</td><td>{impact:.4f}</td></tr>\n"
            elif isinstance(sensitivity, list):
                for item in sensitivity:
                    if isinstance(item, dict):
                        param = item.get("parameter", "")
                        impact = item.get("impact", 0)
                        short = EVIDENCE_SHORT_NAMES.get(param, param)
                        rows += f"<tr><td>{short}</td><td>{impact:.4f}</td></tr>\n"
            if rows:
                sensitivity_section = (
                    '<h2>Sensitivity Analysis</h2>\n'
                    '<table><thead><tr><th>Parameter</th><th>Impact</th></tr></thead>\n'
                    f"<tbody>{rows}</tbody></table>\n"
                )

        recs = _RECOMMENDATIONS.get(result.diagnosis, [])
        rec_items = "".join(f"<li>{r}</li>\n" for r in recs)

        return (
            "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>{title}</title>\n"
            f"<style>{_CSS}</style>\n"
            "</head>\n<body>\n"
            '<div class="container">\n'
            f"<h1>{title}</h1>\n"
            f'<p class="timestamp">{now}</p>\n'
            "<h2>Evidence Summary</h2>\n"
            '<table><thead><tr><th>Parameter</th><th>Value</th><th>Status</th></tr></thead>\n'
            f"<tbody>{evidence_rows}</tbody></table>\n"
            "<h2>Posterior Probabilities</h2>\n"
            f"{bar_rows}\n"
            f'<div class="diagnosis-box {box_class}">\n'
            f"<strong>Diagnosis:</strong> {diag_label} "
            f'<span class="severity-tag" style="background:{diag_color}">'
            f"{result.confidence:.1%}</span>\n"
            f"<br><strong>Confidence:</strong> {result.confidence:.2%}\n"
            f"<br><strong>Investigation required:</strong> "
            f"{'Yes' if result.needs_investigation else 'No'}\n"
            "</div>\n"
            f"{sensitivity_section}\n"
            "<h2>Recommendations</h2>\n"
            f'<ul class="rec">{rec_items}</ul>\n'
            "</div>\n</body>\n</html>"
        )

    def html_comparison_report(
        self,
        results: list[DiagnosisResult],
        labels: list[str] = None,
    ) -> str:
        if labels is None:
            labels = [f"Result {i + 1}" for i in range(len(results))]
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        cards = ""
        for label, result in zip(labels, results):
            diag_label = FAILURE_CLASS_LABELS.get(result.diagnosis, result.diagnosis)
            diag_color = _SEVERITY_COLORS.get(result.diagnosis, "#6b7280")
            box_class = _diagnosis_class(result)

            bar_rows = ""
            for cause in FAILURE_CLASSES:
                prob = result.posteriors.get(cause, 0.0)
                cl = FAILURE_CLASS_LABELS.get(cause, cause)
                color = _SEVERITY_COLORS.get(cause, "#6b7280")
                pct = prob * 100
                bar_rows += (
                    f'<div class="bar-row">'
                    f'<span class="bar-label">{cl}</span>'
                    f'<span class="bar-track">'
                    f'<span class="bar-fill" style="width:{pct:.1f}%;background:{color}"></span>'
                    f'</span>'
                    f'<span class="bar-value">{prob:.1%}</span>'
                    f'</div>\n'
                )

            cards += (
                f'<div style="flex:1;min-width:300px;padding:1rem;border:1px solid #e2e8f0;'
                f'border-radius:6px;margin:0.5rem">\n'
                f"<h3>{label}</h3>\n"
                f"{bar_rows}\n"
                f'<div class="diagnosis-box {box_class}">'
                f"<strong>{diag_label}</strong> "
                f'<span class="severity-tag" style="background:{diag_color}">'
                f"{result.confidence:.1%}</span>\n"
                "</div>\n</div>\n"
            )

        return (
            "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            "<title>Diagnosis Comparison</title>\n"
            f"<style>{_CSS}</style>\n"
            "</head>\n<body>\n"
            '<div class="container">\n'
            "<h1>Diagnosis Comparison</h1>\n"
            f'<p class="timestamp">{now}</p>\n'
            '<div style="display:flex;flex-wrap:wrap;gap:1rem">\n'
            f"{cards}\n</div>\n</div>\n</body>\n</html>"
        )

    def text_report(self, result: DiagnosisResult) -> str:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        lines = [
            "AI CONTROL FAILURE DIAGNOSIS",
            f"Generated: {now}",
            "",
            "Evidence:",
        ]
        ev = result.evidence_vector
        for name in EVIDENCE_PARAMS:
            val = getattr(ev, name)
            if val > 0.0:
                lines.append(f"  {EVIDENCE_SHORT_NAMES[name]}: {val:.2f}")
        if all(getattr(ev, n) == 0.0 for n in EVIDENCE_PARAMS):
            lines.append("  (none)")

        lines.extend(["", "Posterior Probabilities:"])
        for cause, prob in sorted(
            result.posteriors.items(), key=lambda x: x[1], reverse=True
        ):
            label = FAILURE_CLASS_LABELS.get(cause, cause)
            bar = "#" * int(prob * 30)
            lines.append(f"  {label:<35s} {prob:>8.4f}  {bar}")

        diag_label = FAILURE_CLASS_LABELS.get(result.diagnosis, result.diagnosis)
        lines.extend([
            "",
            f"DIAGNOSIS: {diag_label}",
            f"CONFIDENCE: {result.confidence:.2%}",
            f"Investigation required: {'Yes' if result.needs_investigation else 'No'}",
            "",
            "Recommendations:",
        ])
        for rec in _RECOMMENDATIONS.get(result.diagnosis, []):
            lines.append(f"  - {rec}")

        return "\n".join(lines)

    def json_report(self, result: DiagnosisResult) -> str:
        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "evidence": result.evidence_vector.to_dict(),
            **result.to_dict(),
            "recommendations": _RECOMMENDATIONS.get(result.diagnosis, []),
        }
        return json.dumps(data, indent=2)

    def save_report(self, content: str, path: str, format: str = "html") -> None:
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        if format == "json":
            ext = ".json"
        elif format in ("text", "txt"):
            ext = ".txt"
        else:
            ext = ".html"
        if not path.endswith(ext):
            path += ext
        with open(path, "w") as f:
            f.write(content)
