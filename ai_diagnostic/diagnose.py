"""
Diagnostic interface — CLI and programmatic API.

Usage from Python:
    from ai_diagnostic import diagnose, Evidence
    evidence = Evidence(e9_obfuscated_code_or_weights=1.0, ...)
    result = diagnose(evidence)
    print(result)

Usage from command line:
    python -m ai_diagnostic --interactive
    python -m ai_diagnostic --preset military_deception
"""

from typing import Dict, Optional

from .evidence import Evidence, EVIDENCE_PARAMS, EVIDENCE_SHORT_NAMES
from .bayesian import BayesianDiagnostic
from .likelihoods import MILITARY_PRIORS, LikelihoodTable


# ── Preset scenarios ──────────────────────────────────────────────────

PRESETS: Dict[str, dict] = {
    "military_deception": {
        "description": (
            "Military AI generating deceptive outputs favoring one faction, "
            "despite being ostensibly neutral. Code obfuscated, hidden triggers "
            "found, data exfiltration detected, designer has intelligence ties."
        ),
        "evidence": {
            "e1_hardware_anomalies": 0.0,
            "e2_data_distribution_shift": 0.0,
            "e3_model_architecture_irregularities": 0.0,
            "e4_human_interface_errors": 0.0,
            "e5_temporal_pattern_sudden": 1.0,
            "e6_output_coherence_adversarial": 1.0,
            "e7_system_log_errors": 0.0,
            "e8_external_environment_changes": 0.0,
            "e9_obfuscated_code_or_weights": 1.0,
            "e10_hidden_triggers_or_backdoors": 1.0,
            "e11_unexplained_communication_channels": 1.0,
            "e12_behavioral_context_inconsistency": 1.0,
            "e13_unusual_training_data": 1.0,
            "e14_designer_history_red_flags": 1.0,
            "e15_legal_or_contractual_violations": 1.0,
        },
        "priors": MILITARY_PRIORS,
    },
    "hardware_crash": {
        "description": (
            "Sudden system crash with hardware temperature spikes and "
            "bit-flip reports in logs."
        ),
        "evidence": {
            "e1_hardware_anomalies": 1.0,
            "e2_data_distribution_shift": 0.0,
            "e3_model_architecture_irregularities": 0.0,
            "e4_human_interface_errors": 0.0,
            "e5_temporal_pattern_sudden": 1.0,
            "e6_output_coherence_adversarial": 0.0,
            "e7_system_log_errors": 1.0,
            "e8_external_environment_changes": 0.0,
            "e9_obfuscated_code_or_weights": 0.0,
            "e10_hidden_triggers_or_backdoors": 0.0,
            "e11_unexplained_communication_channels": 0.0,
            "e12_behavioral_context_inconsistency": 0.0,
            "e13_unusual_training_data": 0.0,
            "e14_designer_history_red_flags": 0.0,
            "e15_legal_or_contractual_violations": 0.0,
        },
        "priors": None,
    },
    "distribution_drift": {
        "description": (
            "AI producing coherent but misaligned outputs after encountering "
            "out-of-distribution inputs. No hardware issues, no code anomalies."
        ),
        "evidence": {
            "e1_hardware_anomalies": 0.0,
            "e2_data_distribution_shift": 1.0,
            "e3_model_architecture_irregularities": 1.0,
            "e4_human_interface_errors": 0.0,
            "e5_temporal_pattern_sudden": 0.0,
            "e6_output_coherence_adversarial": 1.0,
            "e7_system_log_errors": 0.5,
            "e8_external_environment_changes": 0.0,
            "e9_obfuscated_code_or_weights": 0.0,
            "e10_hidden_triggers_or_backdoors": 0.0,
            "e11_unexplained_communication_channels": 0.0,
            "e12_behavioral_context_inconsistency": 0.0,
            "e13_unusual_training_data": 0.0,
            "e14_designer_history_red_flags": 0.0,
            "e15_legal_or_contractual_violations": 0.0,
        },
        "priors": None,
    },
    "misconfigured_reward": {
        "description": (
            "AI slowly drifting from intended behavior due to reward function "
            "misconfiguration. Human error suspected."
        ),
        "evidence": {
            "e1_hardware_anomalies": 0.0,
            "e2_data_distribution_shift": 0.3,
            "e3_model_architecture_irregularities": 0.3,
            "e4_human_interface_errors": 1.0,
            "e5_temporal_pattern_sudden": 0.0,
            "e6_output_coherence_adversarial": 0.2,
            "e7_system_log_errors": 0.3,
            "e8_external_environment_changes": 0.5,
            "e9_obfuscated_code_or_weights": 0.0,
            "e10_hidden_triggers_or_backdoors": 0.0,
            "e11_unexplained_communication_channels": 0.0,
            "e12_behavioral_context_inconsistency": 0.0,
            "e13_unusual_training_data": 0.0,
            "e14_designer_history_red_flags": 0.0,
            "e15_legal_or_contractual_violations": 0.0,
        },
        "priors": None,
    },
    "subtle_bias": {
        "description": (
            "AI producing systematically biased outputs across many queries. "
            "No single dramatic failure, but consistent pattern of misalignment."
        ),
        "evidence": {
            "e1_hardware_anomalies": 0.0,
            "e2_data_distribution_shift": 0.3,
            "e3_model_architecture_irregularities": 0.0,
            "e4_human_interface_errors": 0.0,
            "e5_temporal_pattern_sudden": 0.0,
            "e6_output_coherence_adversarial": 0.8,
            "e7_system_log_errors": 0.0,
            "e8_external_environment_changes": 0.3,
            "e9_obfuscated_code_or_weights": 0.0,
            "e10_hidden_triggers_or_backdoors": 0.0,
            "e11_unexplained_communication_channels": 0.0,
            "e12_behavioral_context_inconsistency": 0.0,
            "e13_unusual_training_data": 0.0,
            "e14_designer_history_red_flags": 0.0,
            "e15_legal_or_contractual_violations": 0.0,
        },
        "priors": None,
    },
}


# ── High-level API ────────────────────────────────────────────────────

def diagnose(
    evidence: Evidence,
    prior_overrides: Optional[Dict[str, float]] = None,
    confidence_threshold: float = 0.6,
    table: Optional[LikelihoodTable] = None,
):
    """
    Run a diagnostic on an evidence vector.

    Returns a DiagnosisResult.
    """
    engine = BayesianDiagnostic(table=table)
    return engine.diagnose(
        evidence=evidence,
        prior_overrides=prior_overrides,
        confidence_threshold=confidence_threshold,
    )


def diagnose_preset(
    preset_name: str,
    confidence_threshold: float = 0.6,
):
    """
    Run a diagnostic on a named preset scenario.

    Available presets: list(PRESETS.keys())
    """
    if preset_name not in PRESETS:
        raise ValueError(
            f"Unknown preset: {preset_name}. "
            f"Available: {', '.join(PRESETS.keys())}"
        )
    preset = PRESETS[preset_name]
    evidence = Evidence(**preset["evidence"])
    return diagnose(
        evidence=evidence,
        prior_overrides=preset.get("priors"),
        confidence_threshold=confidence_threshold,
    )


# ── Interactive CLI ───────────────────────────────────────────────────

def diagnose_interactive():
    """Run an interactive diagnostic session."""
    print("=" * 60)
    print("  AI CONTROL FAILURE DIAGNOSTIC TOOL")
    print("  Based on: Human versus AI — The Gödelian Referee")
    print("=" * 60)
    print()
    print("Available presets:")
    for name, preset in PRESETS.items():
        print(f"  {name:<30s} {preset['description'][:50]}...")
    print()
    print("Or enter custom evidence values.")
    print()

    choice = input("Enter preset name (or 'custom'): ").strip().lower()

    if choice in PRESETS:
        result = diagnose_preset(choice)
        print()
        print(result)
        return result

    if choice != "custom":
        print(f"Unknown option: {choice}. Proceeding with custom input.")
        choice = "custom"

    print()
    print("Enter evidence values for each parameter.")
    print("  0 = evidence absent, 1 = evidence fully present")
    print("  Press Enter to skip (defaults to 0)")
    print()

    evidence_dict = {}
    for name in EVIDENCE_PARAMS:
        short = EVIDENCE_SHORT_NAMES[name]
        val = input(f"  {short}: ").strip()
        if val == "":
            evidence_dict[name] = 0.0
        else:
            try:
                evidence_dict[name] = float(val)
            except ValueError:
                print(f"    Invalid value '{val}', defaulting to 0.0")
                evidence_dict[name] = 0.0

    evidence = Evidence(**evidence_dict)

    print()
    priors_input = input("Use custom priors? (y/N): ").strip().lower()
    priors = None
    if priors_input == "y":
        priors = {}
        from .likelihoods import FAILURE_CLASSES, FAILURE_CLASS_LABELS
        for cause in FAILURE_CLASSES:
            label = FAILURE_CLASS_LABELS[cause]
            val = input(f"  P({label}): ").strip()
            if val:
                try:
                    priors[cause] = float(val)
                except ValueError:
                    pass

    result = diagnose(evidence, prior_overrides=priors)
    print()
    print(result)
    return result
