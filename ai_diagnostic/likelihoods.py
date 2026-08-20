"""
Likelihood tables for the five failure classes.

Each table maps evidence parameter names to P(e_i | Cause),
the conditional probability of observing that evidence given
a particular failure cause.
"""

from dataclasses import dataclass, field
from typing import Dict

from .evidence import EVIDENCE_PARAMS


FAILURE_CLASSES = [
    "entropy",
    "engineering_limits",
    "human_error",
    "human_bias",
    "human_malice",
]

FAILURE_CLASS_LABELS: Dict[str, str] = {
    "entropy":            "Entropy (physical failure)",
    "engineering_limits": "Engineering Limits (complexity)",
    "human_error":        "Human Error (unintentional)",
    "human_bias":         "Human Bias (systematic)",
    "human_malice":       "Human Malice (deliberate)",
}

# Default priors: P(Cause)
DEFAULT_PRIORS: Dict[str, float] = {
    "entropy":            0.10,
    "engineering_limits": 0.25,
    "human_error":        0.25,
    "human_bias":         0.25,
    "human_malice":       0.15,
}

# Default likelihoods from the paper (Table in Section 9)
# P(e_i | Cause) for each evidence parameter and each cause.
#
# Values are:
#   High   = 0.90 - 0.95
#   Medium = 0.40 - 0.60
#   Low    = 0.01 - 0.10
#
# Continuous evidence uses the same value as binary.

DEFAULT_LIKELIHOODS: Dict[str, Dict[str, float]] = {
    "entropy": {
        "e1_hardware_anomalies":             0.95,
        "e2_data_distribution_shift":        0.05,
        "e3_model_architecture_irregularities": 0.05,
        "e4_human_interface_errors":         0.05,
        "e5_temporal_pattern_sudden":        0.90,
        "e6_output_coherence_adversarial":   0.05,
        "e7_system_log_errors":              0.90,
        "e8_external_environment_changes":   0.05,
        "e9_obfuscated_code_or_weights":     0.01,
        "e10_hidden_triggers_or_backdoors":  0.01,
        "e11_unexplained_communication_channels": 0.01,
        "e12_behavioral_context_inconsistency": 0.01,
        "e13_unusual_training_data":         0.01,
        "e14_designer_history_red_flags":    0.01,
        "e15_legal_or_contractual_violations": 0.01,
    },
    "engineering_limits": {
        "e1_hardware_anomalies":             0.05,
        "e2_data_distribution_shift":        0.90,
        "e3_model_architecture_irregularities": 0.90,
        "e4_human_interface_errors":         0.05,
        "e5_temporal_pattern_sudden":        0.40,
        "e6_output_coherence_adversarial":   0.80,
        "e7_system_log_errors":              0.50,
        "e8_external_environment_changes":   0.10,
        "e9_obfuscated_code_or_weights":     0.05,
        "e10_hidden_triggers_or_backdoors":  0.05,
        "e11_unexplained_communication_channels": 0.05,
        "e12_behavioral_context_inconsistency": 0.10,
        "e13_unusual_training_data":         0.10,
        "e14_designer_history_red_flags":    0.05,
        "e15_legal_or_contractual_violations": 0.05,
    },
    "human_error": {
        "e1_hardware_anomalies":             0.05,
        "e2_data_distribution_shift":        0.40,
        "e3_model_architecture_irregularities": 0.30,
        "e4_human_interface_errors":         0.95,
        "e5_temporal_pattern_sudden":        0.30,
        "e6_output_coherence_adversarial":   0.20,
        "e7_system_log_errors":              0.40,
        "e8_external_environment_changes":   0.80,
        "e9_obfuscated_code_or_weights":     0.02,
        "e10_hidden_triggers_or_backdoors":  0.02,
        "e11_unexplained_communication_channels": 0.01,
        "e12_behavioral_context_inconsistency": 0.05,
        "e13_unusual_training_data":         0.05,
        "e14_designer_history_red_flags":    0.02,
        "e15_legal_or_contractual_violations": 0.02,
    },
    "human_bias": {
        "e1_hardware_anomalies":             0.03,
        "e2_data_distribution_shift":        0.40,
        "e3_model_architecture_irregularities": 0.30,
        "e4_human_interface_errors":         0.50,
        "e5_temporal_pattern_sudden":        0.20,
        "e6_output_coherence_adversarial":   0.80,
        "e7_system_log_errors":              0.10,
        "e8_external_environment_changes":   0.40,
        "e9_obfuscated_code_or_weights":     0.03,
        "e10_hidden_triggers_or_backdoors":  0.03,
        "e11_unexplained_communication_channels": 0.02,
        "e12_behavioral_context_inconsistency": 0.08,
        "e13_unusual_training_data":         0.08,
        "e14_designer_history_red_flags":    0.03,
        "e15_legal_or_contractual_violations": 0.03,
    },
    "human_malice": {
        "e1_hardware_anomalies":             0.01,
        "e2_data_distribution_shift":        0.05,
        "e3_model_architecture_irregularities": 0.05,
        "e4_human_interface_errors":         0.05,
        "e5_temporal_pattern_sudden":        0.40,
        "e6_output_coherence_adversarial":   0.60,
        "e7_system_log_errors":              0.10,
        "e8_external_environment_changes":   0.05,
        "e9_obfuscated_code_or_weights":     0.95,
        "e10_hidden_triggers_or_backdoors":  0.95,
        "e11_unexplained_communication_channels": 0.90,
        "e12_behavioral_context_inconsistency": 0.90,
        "e13_unusual_training_data":         0.90,
        "e14_designer_history_red_flags":    0.80,
        "e15_legal_or_contractual_violations": 0.80,
    },
}


@dataclass
class LikelihoodTable:
    """
    A complete likelihood table mapping evidence parameters to
    conditional probabilities for each failure class.
    """
    priors: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_PRIORS))
    likelihoods: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {k: dict(v) for k, v in DEFAULT_LIKELIHOODS.items()}
    )

    def p_evidence_given_cause(self, cause: str, evidence_name: str) -> float:
        """Return P(e_i | Cause)."""
        return self.likelihoods[cause][evidence_name]

    def prior(self, cause: str) -> float:
        """Return P(Cause)."""
        return self.priors[cause]

    def set_prior(self, cause: str, value: float) -> None:
        """Set P(Cause)."""
        if cause not in self.priors:
            raise ValueError(f"Unknown cause: {cause}")
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"Prior must be in [0, 1], got {value}")
        self.priors[cause] = value

    def set_likelihood(self, cause: str, evidence_name: str, value: float) -> None:
        """Set P(e_i | Cause)."""
        if cause not in self.likelihoods:
            raise ValueError(f"Unknown cause: {cause}")
        if evidence_name not in self.likelihoods[cause]:
            raise ValueError(f"Unknown evidence: {evidence_name}")
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"Likelihood must be in [0, 1], got {value}")
        self.likelihoods[cause][evidence_name] = value

    def validate(self) -> list[str]:
        """Check table for common issues. Return list of warnings."""
        warnings = []
        prior_sum = sum(self.priors.values())
        if abs(prior_sum - 1.0) > 0.01:
            warnings.append(f"Priors sum to {prior_sum:.4f}, expected 1.0")

        for cause in FAILURE_CLASSES:
            for evid in EVIDENCE_PARAMS:
                val = self.likelihoods[cause][evid]
                if val == 0.0:
                    warnings.append(
                        f"P({evid} | {cause}) = 0.0 — will zero out entire posterior. "
                        f"Consider using a small floor value (e.g., 0.01)."
                    )
        return warnings


# Military context: adjusted priors
MILITARY_PRIORS = dict(DEFAULT_PRIORS)
MILITARY_PRIORS["human_malice"] = 0.15
MILITARY_PRIORS["entropy"] = 0.01
MILITARY_PRIORS["engineering_limits"] = 0.05
MILITARY_PRIORS["human_error"] = 0.05
MILITARY_PRIORS["human_bias"] = 0.05
