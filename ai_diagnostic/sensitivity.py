"""
Sensitivity analysis for the Bayesian diagnostic framework.

Provides four analyses:
  1. Parameter sensitivity — how much each evidence parameter contributes
  2. Flip analysis — which missing evidence, if found, would change the diagnosis
  3. Tipping point — minimum evidence value at which the diagnosis flips
  4. Prior sensitivity — effect of perturbing priors ±10%
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .evidence import Evidence, EVIDENCE_PARAMS
from .bayesian import BayesianDiagnostic, DiagnosisResult
from .likelihoods import LikelihoodTable, FAILURE_CLASSES


@dataclass
class SensitivityReport:
    """Results of a full sensitivity analysis."""

    parameter_sensitivity: Dict[str, float] = field(default_factory=dict)
    flip_analysis: List[Dict[str, object]] = field(default_factory=list)
    tipping_points: Dict[str, Optional[float]] = field(default_factory=dict)
    prior_sensitivity: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  SENSITIVITY ANALYSIS",
            "=" * 60,
        ]

        lines.append("")
        lines.append("-" * 60)
        lines.append("  Parameter Sensitivity (|delta in top posterior|)")
        lines.append("-" * 60)
        if self.parameter_sensitivity:
            sorted_params = sorted(
                self.parameter_sensitivity.items(), key=lambda x: abs(x[1]), reverse=True
            )
            for name, delta in sorted_params:
                lines.append(f"  {name:<50s} {delta:>+8.4f}")
        else:
            lines.append("  (none)")

        lines.append("")
        lines.append("-" * 60)
        lines.append("  Flip Analysis (missing evidence that would change diagnosis)")
        lines.append("-" * 60)
        if self.flip_analysis:
            for entry in self.flip_analysis:
                lines.append(
                    f"  {entry['parameter']:<40s} "
                    f"→ {entry['alternative_diagnosis']:<25s} "
                    f"conf={entry['new_confidence']:.2%}"
                )
        else:
            lines.append("  (no missing evidence that would flip diagnosis)")

        lines.append("")
        lines.append("-" * 60)
        lines.append("  Tipping Points (evidence value at which diagnosis flips)")
        lines.append("-" * 60)
        if self.tipping_points:
            for name, val in self.tipping_points.items():
                if val is None:
                    lines.append(f"  {name:<50s}   (no flip in [0, 1])")
                else:
                    lines.append(f"  {name:<50s}   {val:.4f}")
        else:
            lines.append("  (none)")

        lines.append("")
        lines.append("-" * 60)
        lines.append("  Prior Sensitivity (|delta| for ±10% perturbation)")
        lines.append("-" * 60)
        if self.prior_sensitivity:
            sorted_priors = sorted(
                self.prior_sensitivity.items(), key=lambda x: abs(x[1]), reverse=True
            )
            for cause, delta in sorted_priors:
                lines.append(f"  {cause:<50s} {delta:>+8.4f}")
        else:
            lines.append("  (none)")

        lines.append("=" * 60)
        return "\n".join(lines)


def _rediagnose(
    diagnostic: BayesianDiagnostic,
    evidence: Evidence,
    prior_overrides: Optional[Dict[str, float]] = None,
) -> DiagnosisResult:
    return diagnostic.diagnose(evidence, prior_overrides=prior_overrides)


def _compute_parameter_sensitivity(
    result: DiagnosisResult,
    diagnostic: BayesianDiagnostic,
) -> Dict[str, float]:
    """For each evidence param, set to 0 and measure change in top posterior."""
    top_diagnosis = result.diagnosis
    base_confidence = result.confidence
    evidence = result.evidence_vector
    sensitivity: Dict[str, float] = {}

    for param in EVIDENCE_PARAMS:
        current_val = getattr(evidence, param)
        if current_val == 0.0:
            sensitivity[param] = 0.0
            continue
        zeroed = Evidence(**{**evidence.to_dict(), param: 0.0})
        new_result = _rediagnose(diagnostic, zeroed)
        sensitivity[param] = new_result.posteriors.get(top_diagnosis, 0.0) - base_confidence

    return sensitivity


def _compute_flip_analysis(
    result: DiagnosisResult,
    diagnostic: BayesianDiagnostic,
) -> List[Dict[str, object]]:
    """For each evidence param at 0, set to 1 and check if diagnosis changes."""
    evidence = result.evidence_vector
    current_diagnosis = result.diagnosis
    flips: List[Dict[str, object]] = []

    for param in EVIDENCE_PARAMS:
        current_val = getattr(evidence, param)
        if current_val != 0.0:
            continue
        one_val = Evidence(**{**evidence.to_dict(), param: 1.0})
        new_result = _rediagnose(diagnostic, one_val)
        if new_result.diagnosis != current_diagnosis:
            flips.append({
                "parameter": param,
                "current_value": current_val,
                "alternative_diagnosis": new_result.diagnosis,
                "new_confidence": new_result.confidence,
            })

    return flips


def _compute_tipping_points(
    result: DiagnosisResult,
    diagnostic: BayesianDiagnostic,
) -> Dict[str, Optional[float]]:
    """For each evidence param, find minimum value at which diagnosis flips."""
    top_diagnosis = result.diagnosis
    evidence = result.evidence_vector
    tipping: Dict[str, Optional[float]] = {}

    for param in EVIDENCE_PARAMS:
        current_val = getattr(evidence, param)
        if current_val >= 1.0:
            tipping[param] = None
            continue

        lo, hi = current_val, 1.0
        found = None

        for _ in range(50):
            mid = (lo + hi) / 2.0
            probe = Evidence(**{**evidence.to_dict(), param: mid})
            probe_result = _rediagnose(diagnostic, probe)
            if probe_result.diagnosis != top_diagnosis:
                found = mid
                hi = mid
            else:
                lo = mid
            if hi - lo < 1e-6:
                break

        tipping[param] = found

    return tipping


def _compute_prior_sensitivity(
    result: DiagnosisResult,
    diagnostic: BayesianDiagnostic,
) -> Dict[str, float]:
    """Perturb each prior ±10% and measure change in top posterior."""
    top_diagnosis = result.diagnosis
    base_confidence = result.confidence
    evidence = result.evidence_vector
    sensitivity: Dict[str, float] = {}

    for cause in FAILURE_CLASSES:
        base_prior = diagnostic.table.prior(cause)
        perturbed_prior = min(base_prior * 1.1, 1.0)
        overrides = {cause: perturbed_prior}
        up_result = _rediagnose(diagnostic, evidence, prior_overrides=overrides)
        delta = up_result.posteriors.get(top_diagnosis, 0.0) - base_confidence
        sensitivity[cause] = delta

    return sensitivity


def compute_sensitivity(
    result: DiagnosisResult,
    diagnostic: BayesianDiagnostic,
) -> SensitivityReport:
    """
    Compute all four sensitivity analyses for a given diagnosis.

    Args:
        result: A completed DiagnosisResult.
        diagnostic: The BayesianDiagnostic engine used to produce it.

    Returns:
        A SensitivityReport with parameter sensitivity, flip analysis,
        tipping points, and prior sensitivity.
    """
    return SensitivityReport(
        parameter_sensitivity=_compute_parameter_sensitivity(result, diagnostic),
        flip_analysis=_compute_flip_analysis(result, diagnostic),
        tipping_points=_compute_tipping_points(result, diagnostic),
        prior_sensitivity=_compute_prior_sensitivity(result, diagnostic),
    )
