"""
AI Diagnostic Tool — Comprehensive CLI

Usage:
    python -m ai_diagnostic diagnose --preset military_deception
    python -m ai_diagnostic diagnose --interactive
    python -m ai_diagnostic batch --input cases.json --output results.json
    python -m ai_diagnostic sensitivity --preset military_deception
    python -m ai_diagnostic compare --preset1 hardware_crash --preset2 distribution_drift
    python -m ai_diagnostic report --preset military_deception --format html --output report.html
    python -m ai_diagnostic history --recent 10
    python -m ai_diagnostic serve --port 8000
    python -m ai_diagnostic presets
    python -m ai_diagnostic params
"""

import argparse
import json
import sys
from datetime import datetime

from .diagnose import diagnose, diagnose_preset, PRESETS
from .evidence import Evidence, EVIDENCE_PARAMS, EVIDENCE_SHORT_NAMES, EVIDENCE_DESCRIPTIONS
from .sensitivity import compute_sensitivity
from .bayesian import BayesianDiagnostic
from .history import DiagnosisHistory
from .batch import BatchProcessor
from .reports import ReportGenerator
from .likelihoods import FAILURE_CLASS_LABELS


def cmd_diagnose(args):
    """Run a single diagnosis."""
    if args.interactive:
        from .diagnose import diagnose_interactive
        return diagnose_interactive()

    if args.preset:
        result = diagnose_preset(args.preset)
    elif args.evidence_file:
        with open(args.evidence_file) as f:
            data = json.load(f)
        evidence = Evidence(**data.get("evidence", data))
        priors = data.get("priors")
        result = diagnose(evidence, prior_overrides=priors)
    else:
        print("Error: specify --preset, --evidence-file, or --interactive", file=sys.stderr)
        sys.exit(1)

    print(result)

    if args.sensitivity:
        engine = BayesianDiagnostic()
        sens = compute_sensitivity(result, engine)
        print()
        print(sens.summary())

    if args.history:
        history = DiagnosisHistory()
        evidence = result.evidence_vector
        history.record(result, evidence, metadata={"source": "cli"})
        print(f"\nRecorded to history (id={len(history.recent(1))})")


def cmd_sensitivity(args):
    """Run sensitivity analysis."""
    if args.preset:
        result = diagnose_preset(args.preset)
    else:
        print("Error: specify --preset", file=sys.stderr)
        sys.exit(1)

    engine = BayesianDiagnostic()
    sens = compute_sensitivity(result, engine)
    print(sens.summary())


def cmd_compare(args):
    """Compare two scenarios side by side."""
    if args.preset1 and args.preset2:
        r1 = diagnose_preset(args.preset1)
        r2 = diagnose_preset(args.preset2)
    else:
        print("Error: specify --preset1 and --preset2", file=sys.stderr)
        sys.exit(1)

    bp = BatchProcessor()
    print(bp.compare([r1, r2], labels=[args.preset1, args.preset2]))


def cmd_batch(args):
    """Batch process multiple cases."""
    with open(args.input) as f:
        data = json.load(f)

    cases = data if isinstance(data, list) else data.get("cases", data.get("evidence", []))
    evidence_list = [Evidence(**c) for c in cases]

    bp = BatchProcessor()
    results = bp.process(evidence_list)
    summary = bp.summary(results)

    print(f"Processed {len(results)} cases")
    print(f"Diagnoses: {summary['diagnosis_counts']}")
    print(f"Average confidence: {summary['average_confidence']:.2%}")
    print(f"Flagged for investigation: {summary['flagged_count']}")

    if args.output:
        bp.export_batch(results, args.output, format=args.format)
        print(f"Exported to {args.output}")


def cmd_report(args):
    """Generate a report."""
    if args.preset:
        result = diagnose_preset(args.preset)
    else:
        print("Error: specify --preset", file=sys.stderr)
        sys.exit(1)

    gen = ReportGenerator()

    if args.format == "html":
        engine = BayesianDiagnostic()
        sens = compute_sensitivity(result, engine)
        content = gen.html_report(result, sensitivity=sens)
        ext = "html"
    elif args.format == "json":
        content = gen.json_report(result)
        ext = "json"
    else:
        content = gen.text_report(result)
        ext = "txt"

    if args.output:
        gen.save_report(content, args.output)
        print(f"Report saved to {args.output}")
    else:
        print(content)


def cmd_history(args):
    """View diagnosis history."""
    history = DiagnosisHistory()

    if args.recent:
        records = history.recent(args.recent)
    elif args.type_filter:
        records = history.by_diagnosis(args.type_filter)
    else:
        records = history.recent(20)

    if not records:
        print("No records found.")
        return

    if args.stats:
        stats = history.stats()
        print(json.dumps(stats, indent=2, default=str))
        return

    print(f"{'ID':<6} {'Timestamp':<20} {'Diagnosis':<25} {'Confidence':<12}")
    print("-" * 65)
    for r in records:
        ts = r["timestamp"][:19] if r["timestamp"] else "N/A"
        print(f"{r['id']:<6} {ts:<20} {r['diagnosis']:<25} {r['confidence']:<12.2%}")

    if args.export_csv:
        history.export_csv(args.export_csv)
        print(f"Exported to {args.export_csv}")


def cmd_presets(args):
    """List available presets."""
    print("Available presets:\n")
    for name, preset in PRESETS.items():
        print(f"  {name}")
        print(f"    {preset['description']}")
        evid_count = sum(1 for v in preset["evidence"].values() if v > 0)
        print(f"    Evidence parameters active: {evid_count}/15")
        print()


def cmd_params(args):
    """List evidence parameters."""
    print("Evidence Parameters:\n")
    for name in EVIDENCE_PARAMS:
        short = EVIDENCE_SHORT_NAMES[name]
        desc = EVIDENCE_DESCRIPTIONS[name]
        print(f"  {name}")
        print(f"    Short: {short}")
        print(f"    Desc:  {desc}")
        print()


def cmd_serve(args):
    """Start the REST API server."""
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn not installed. Run: pip install uvicorn", file=sys.stderr)
        sys.exit(1)

    from .api import create_app
    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port)


def main():
    parser = argparse.ArgumentParser(
        prog="ai_diagnostic",
        description="AI Control Failure Diagnostic Framework",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # diagnose
    p_diag = subparsers.add_parser("diagnose", help="Run a diagnosis")
    p_diag.add_argument("--preset", choices=list(PRESETS.keys()))
    p_diag.add_argument("--evidence-file", help="JSON file with evidence parameters")
    p_diag.add_argument("--interactive", action="store_true")
    p_diag.add_argument("--sensitivity", "-s", action="store_true", help="Include sensitivity analysis")
    p_diag.add_argument("--history", action="store_true", help="Record to history")
    p_diag.set_defaults(func=cmd_diagnose)

    # sensitivity
    p_sens = subparsers.add_parser("sensitivity", help="Run sensitivity analysis")
    p_sens.add_argument("--preset", choices=list(PRESETS.keys()), required=True)
    p_sens.set_defaults(func=cmd_sensitivity)

    # compare
    p_comp = subparsers.add_parser("compare", help="Compare two scenarios")
    p_comp.add_argument("--preset1", choices=list(PRESETS.keys()), required=True)
    p_comp.add_argument("--preset2", choices=list(PRESETS.keys()), required=True)
    p_comp.set_defaults(func=cmd_compare)

    # batch
    p_batch = subparsers.add_parser("batch", help="Batch process cases")
    p_batch.add_argument("--input", "-i", required=True, help="JSON file with list of cases")
    p_batch.add_argument("--output", "-o", help="Export results to file")
    p_batch.add_argument("--format", choices=["csv", "json"], default="json")
    p_batch.set_defaults(func=cmd_batch)

    # report
    p_report = subparsers.add_parser("report", help="Generate a report")
    p_report.add_argument("--preset", choices=list(PRESETS.keys()), required=True)
    p_report.add_argument("--format", choices=["html", "json", "text"], default="html")
    p_report.add_argument("--output", "-o", help="Save to file")
    p_report.set_defaults(func=cmd_report)

    # history
    p_hist = subparsers.add_parser("history", help="View diagnosis history")
    p_hist.add_argument("--recent", "-n", type=int, help="Show last N records")
    p_hist.add_argument("--type-filter", help="Filter by diagnosis type")
    p_hist.add_argument("--stats", action="store_true", help="Show statistics")
    p_hist.add_argument("--export-csv", help="Export to CSV")
    p_hist.set_defaults(func=cmd_history)

    # presets
    p_presets = subparsers.add_parser("presets", help="List available presets")
    p_presets.set_defaults(func=cmd_presets)

    # params
    p_params = subparsers.add_parser("params", help="List evidence parameters")
    p_params.set_defaults(func=cmd_params)

    # serve
    p_serve = subparsers.add_parser("serve", help="Start REST API server")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
