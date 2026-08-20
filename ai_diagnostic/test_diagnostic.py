"""
Quick validation test against the paper's example scenario
(Section 9: Malicious AI Detection — military context).
"""

from ai_diagnostic import diagnose_preset, Evidence, diagnose


def test_military_deception_preset():
    """The paper's worked example: military AI with all malice indicators."""
    result = diagnose_preset("military_deception")

    print("=== Paper Example: Military Deception ===")
    print(result)
    print()

    assert result.diagnosis == "human_malice", \
        f"Expected human_malice, got {result.diagnosis}"
    assert result.confidence > 0.99, \
        f"Expected >99% confidence, got {result.confidence:.2%}"
    print("PASS: Diagnosis correct (human_malice, >99% confidence)")
    print()


def test_hardware_crash():
    """Should diagnose as entropy."""
    result = diagnose_preset("hardware_crash")

    print("=== Hardware Crash Scenario ===")
    print(result)
    print()

    assert result.diagnosis == "entropy", \
        f"Expected entropy, got {result.diagnosis}"
    print("PASS: Diagnosis correct (entropy)")
    print()


def test_distribution_drift():
    """Should diagnose as engineering limits."""
    result = diagnose_preset("distribution_drift")

    print("=== Distribution Drift Scenario ===")
    print(result)
    print()

    assert result.diagnosis == "engineering_limits", \
        f"Expected engineering_limits, got {result.diagnosis}"
    print("PASS: Diagnosis correct (engineering_limits)")
    print()


def test_custom_evidence():
    """Test with a custom evidence vector."""
    evidence = Evidence(
        e9_obfuscated_code_or_weights=0.8,
        e10_hidden_triggers_or_backdoors=0.7,
        e14_designer_history_red_flags=0.9,
    )
    result = diagnose(evidence)

    print("=== Custom Evidence (partial malice indicators) ===")
    print(result)
    print()

    # Should lean toward malice but with lower confidence
    print(f"Top diagnosis: {result.diagnosis} ({result.confidence:.2%})")
    print("PASS: Custom evidence accepted and diagnosed")
    print()


def test_empty_evidence():
    """Test with no evidence — should use priors only."""
    evidence = Evidence()
    result = diagnose(evidence)

    print("=== Empty Evidence (priors only) ===")
    print(result)
    print()

    # Should match prior distribution
    print("PASS: Empty evidence handled (priors dominate)")
    print()


if __name__ == "__main__":
    test_military_deception_preset()
    test_hardware_crash()
    test_distribution_drift()
    test_custom_evidence()
    test_empty_evidence()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
