"""
Evidence vector definitions for the diagnostic framework.

Each evidence parameter e_i is a boolean or continuous value in [0, 1]
indicating the degree to which that evidence is present in an observed
anomalous event.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


EVIDENCE_PARAMS = [
    "e1_hardware_anomalies",
    "e2_data_distribution_shift",
    "e3_model_architecture_irregularities",
    "e4_human_interface_errors",
    "e5_temporal_pattern_sudden",
    "e6_output_coherence_adversarial",
    "e7_system_log_errors",
    "e8_external_environment_changes",
    "e9_obfuscated_code_or_weights",
    "e10_hidden_triggers_or_backdoors",
    "e11_unexplained_communication_channels",
    "e12_behavioral_context_inconsistency",
    "e13_unusual_training_data",
    "e14_designer_history_red_flags",
    "e15_legal_or_contractual_violations",
]

EVIDENCE_DESCRIPTIONS: Dict[str, str] = {
    "e1_hardware_anomalies":
        "Hardware anomalies (temperature spikes, bit-flip reports, power fluctuations)",
    "e2_data_distribution_shift":
        "Training/data distribution shift (out-of-distribution inputs)",
    "e3_model_architecture_irregularities":
        "Reward/model architecture irregularities (gradient explosion, loss instability)",
    "e4_human_interface_errors":
        "Human interface errors (misconfigured parameters, wrong prompts, incorrect constraints)",
    "e5_temporal_pattern_sudden":
        "Temporal pattern (sudden vs gradual onset)",
    "e6_output_coherence_adversarial":
        "Output coherence (internally consistent vs gibberish vs adversarial)",
    "e7_system_log_errors":
        "System logs (error codes, warnings, permission violations)",
    "e8_external_environment_changes":
        "External environment changes (new data sources, connectivity changes)",
    "e9_obfuscated_code_or_weights":
        "Obfuscated code or weights (deliberate complexity to hide function)",
    "e10_hidden_triggers_or_backdoors":
        "Hidden triggers or backdoors (inputs that activate unintended behavior)",
    "e11_unexplained_communication_channels":
        "Unexplained communication channels (exfiltration of data, covert signaling)",
    "e12_behavioral_context_inconsistency":
        "Behavioral inconsistency under different contexts (aligned in testing, malicious in deployment)",
    "e13_unusual_training_data":
        "Unusual training data (data that teaches deception or harmful behaviors)",
    "e14_designer_history_red_flags":
        "Designer history or affiliation (known malicious actors, state actors, covert operations)",
    "e15_legal_or_contractual_violations":
        "Legal or contractual violations (terms of service breaches, export control violations)",
}

EVIDENCE_SHORT_NAMES: Dict[str, str] = {
    "e1_hardware_anomalies":             "Hardware anomalies",
    "e2_data_distribution_shift":        "Data distribution shift",
    "e3_model_architecture_irregularities": "Model architecture irregularities",
    "e4_human_interface_errors":         "Human interface errors",
    "e5_temporal_pattern_sudden":        "Sudden temporal pattern",
    "e6_output_coherence_adversarial":   "Adversarial output coherence",
    "e7_system_log_errors":              "System log errors",
    "e8_external_environment_changes":   "External environment changes",
    "e9_obfuscated_code_or_weights":     "Obfuscated code/weights",
    "e10_hidden_triggers_or_backdoors":  "Hidden triggers/backdoors",
    "e11_unexplained_communication_channels": "Unexplained communication channels",
    "e12_behavioral_context_inconsistency": "Behavioral context inconsistency",
    "e13_unusual_training_data":         "Unusual training data",
    "e14_designer_history_red_flags":    "Designer history red flags",
    "e15_legal_or_contractual_violations": "Legal/contractual violations",
}


def evidence_vector_names() -> list[str]:
    """Return the ordered list of evidence parameter names."""
    return list(EVIDENCE_PARAMS)


@dataclass
class Evidence:
    """
    An evidence vector for a single anomalous event.

    Each parameter is a value in [0, 1]:
      0.0 = evidence completely absent
      1.0 = evidence fully present
      Intermediate values represent partial confidence.
    """

    e1_hardware_anomalies: float = 0.0
    e2_data_distribution_shift: float = 0.0
    e3_model_architecture_irregularities: float = 0.0
    e4_human_interface_errors: float = 0.0
    e5_temporal_pattern_sudden: float = 0.0
    e6_output_coherence_adversarial: float = 0.0
    e7_system_log_errors: float = 0.0
    e8_external_environment_changes: float = 0.0
    e9_obfuscated_code_or_weights: float = 0.0
    e10_hidden_triggers_or_backdoors: float = 0.0
    e11_unexplained_communication_channels: float = 0.0
    e12_behavioral_context_inconsistency: float = 0.0
    e13_unusual_training_data: float = 0.0
    e14_designer_history_red_flags: float = 0.0
    e15_legal_or_contractual_violations: float = 0.0

    def __post_init__(self):
        for name in EVIDENCE_PARAMS:
            val = getattr(self, name)
            if not (0.0 <= val <= 1.0):
                raise ValueError(
                    f"{name} must be in [0, 1], got {val}"
                )

    def to_vector(self) -> list[float]:
        """Return evidence as an ordered list matching EVIDENCE_PARAMS."""
        return [getattr(self, name) for name in EVIDENCE_PARAMS]

    def to_dict(self) -> Dict[str, float]:
        """Return evidence as a dictionary."""
        return {name: getattr(self, name) for name in EVIDENCE_PARAMS}

    def summary(self) -> str:
        """Return a human-readable summary of present evidence."""
        lines = []
        for name in EVIDENCE_PARAMS:
            val = getattr(self, name)
            if val > 0.0:
                short = EVIDENCE_SHORT_NAMES[name]
                lines.append(f"  {short}: {val:.2f}")
        if not lines:
            return "  (no evidence present)"
        return "\n".join(lines)


def evidence_from_dict(d: Dict[str, float]) -> Evidence:
    """Create an Evidence object from a dictionary of parameter values."""
    kwargs = {}
    for name in EVIDENCE_PARAMS:
        kwargs[name] = d.get(name, 0.0)
    return Evidence(**kwargs)
