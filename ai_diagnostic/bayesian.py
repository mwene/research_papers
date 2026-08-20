"""
Bayesian diagnostic engine.

Computes posterior probabilities P(Cause | evidence) using the
likelihood tables and evidence vectors defined in the paper.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .evidence import Evidence, EVIDENCE_PARAMS
from .likelihoods import (
    LikelihoodTable,
    FAILURE_CLASSES,
    FAILURE_CLASS_LABELS,
)


@dataclass
class DiagnosisResult:
    """Result of a Bayesian diagnostic computation."""

    posteriors: Dict[str, float]
    evidence_vector: Evidence
    diagnosis: str
    confidence: float
    needs_investigation: bool
    log_likelihoods: Dict[str, float]

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "  AI CONTROL FAILURE DIAGNOSIS",
            "=" * 60,
            "",
            "Evidence present:",
            self.evidence_vector.summary(),
            "",
            "-" * 60,
            "  Posterior Probabilities",
            "-" * 60,
        ]
        for cause in sorted(self.posteriors, key=self.posteriors.get, reverse=True):
            prob = self.posteriors[cause]
            label = FAILURE_CLASS_LABELS.get(cause, cause)
            bar = "#" * int(prob * 40)
            lines.append(f"  {label:<35s} {prob:>8.4f}  {bar}")

        lines.extend([
            "",
            "-" * 60,
            f"  DIAGNOSIS: {FAILURE_CLASS_LABELS.get(self.diagnosis, self.diagnosis)}",
            f"  CONFIDENCE: {self.confidence:.2%}",
        ])
        if self.needs_investigation:
            lines.append(
                "  ⚠ Confidence below threshold — further investigation recommended."
            )
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Return results as a serializable dictionary."""
        return {
            "posteriors": self.posteriors,
            "diagnosis": self.diagnosis,
            "diagnosis_label": FAILURE_CLASS_LABELS.get(self.diagnosis, self.diagnosis),
            "confidence": self.confidence,
            "needs_investigation": self.needs_investigation,
            "log_likelihoods": self.log_likelihoods,
        }


class BayesianDiagnostic:
    """
    Bayesian diagnostic engine for AI control failures.

    Usage:
        diag = BayesianDiagnostic()
        result = diag.diagnose(evidence, prior_overrides=None)
        print(result)
    """

    def __init__(self, table: Optional[LikelihoodTable] = None):
        self.table = table or LikelihoodTable()

    def diagnose(
        self,
        evidence: Evidence,
        prior_overrides: Optional[Dict[str, float]] = None,
        confidence_threshold: float = 0.6,
    ) -> DiagnosisResult:
        """
        Compute posterior probabilities and return a diagnosis.

        Args:
            evidence: The observed evidence vector.
            prior_overrides: Optional dict overriding P(Cause) values.
            confidence_threshold: Minimum confidence to avoid flagging
                                  for further investigation.

        Returns:
            DiagnosisResult with posteriors, diagnosis, and confidence.
        """
        # Apply prior overrides
        table = self.table
        if prior_overrides:
            table = LikelihoodTable(
                priors={**self.table.priors, **prior_overrides},
                likelihoods=self.table.likelihoods,
            )

        # Compute log P(e | Cause) for numerical stability
        evidence_vec = evidence.to_vector()
        log_likelihoods: Dict[str, float] = {}
        log_priors: Dict[str, float] = {}

        for cause in FAILURE_CLASSES:
            log_priors[cause] = math.log(max(table.prior(cause), 1e-300))
            log_lik = 0.0
            for i, evid_name in enumerate(EVIDENCE_PARAMS):
                ei = evidence_vec[i]
                pe_c = table.p_evidence_given_cause(cause, evid_name)

                # For binary evidence (0 or 1):
                #   P(e=1 | C) = pe_c
                #   P(e=0 | C) = 1 - pe_c
                # For continuous evidence in (0,1):
                #   Use pe_c^ei * (1-pe_c)^(1-ei) as a interpolation
                if ei == 0.0:
                    log_lik += math.log(max(1.0 - pe_c, 1e-300))
                elif ei == 1.0:
                    log_lik += math.log(max(pe_c, 1e-300))
                else:
                    # Continuous: weighted geometric mean
                    log_lik += ei * math.log(max(pe_c, 1e-300)) + \
                               (1.0 - ei) * math.log(max(1.0 - pe_c, 1e-300))

            log_likelihoods[cause] = log_priors[cause] + log_lik

        # Normalize using log-sum-exp trick
        max_log = max(log_likelihoods.values())
        exp_shifted = {
            cause: math.exp(log_val - max_log)
            for cause, log_val in log_likelihoods.items()
        }
        total = sum(exp_shifted.values())
        posteriors = {
            cause: val / total for cause, val in exp_shifted.items()
        }

        # Determine diagnosis
        diagnosis = max(posteriors, key=posteriors.get)
        confidence = posteriors[diagnosis]
        needs_investigation = confidence < confidence_threshold

        return DiagnosisResult(
            posteriors=posteriors,
            evidence_vector=evidence,
            diagnosis=diagnosis,
            confidence=confidence,
            needs_investigation=needs_investigation,
            log_likelihoods=log_likelihoods,
        )
