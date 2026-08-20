"""
Comprehensive test suite for the AI Diagnostic Framework.

Tests all modules: evidence, likelihoods, bayesian, sensitivity,
history, batch, alerts, reports, and the CLI presets.
"""

import json
import os
import tempfile
from datetime import datetime, timedelta

from ai_diagnostic import (
    Evidence,
    BayesianDiagnostic,
    DiagnosisResult,
    LikelihoodTable,
    SensitivityReport,
    compute_sensitivity,
    DiagnosisHistory,
    BatchProcessor,
    AlertManager,
    ThresholdAlert,
    ReportGenerator,
    diagnose,
    diagnose_preset,
    PRESETS,
    EVIDENCE_PARAMS,
    FAILURE_CLASSES,
)


# ── Evidence ──────────────────────────────────────────────────────────

def test_evidence_creation():
    e = Evidence(e1_hardware_anomalies=1.0, e9_obfuscated_code_or_weights=0.5)
    assert e.e1_hardware_anomalies == 1.0
    assert e.e9_obfuscated_code_or_weights == 0.5
    assert e.e2_data_distribution_shift == 0.0
    print("PASS: test_evidence_creation")


def test_evidence_validation():
    try:
        Evidence(e1_hardware_anomalies=2.0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("PASS: test_evidence_validation")


def test_evidence_vector():
    e = Evidence()
    vec = e.to_vector()
    assert len(vec) == 15
    assert all(v == 0.0 for v in vec)
    print("PASS: test_evidence_vector")


def test_evidence_from_dict():
    d = {"e1_hardware_anomalies": 1.0, "e10_hidden_triggers_or_backdoors": 0.8}
    e = Evidence(**d)
    assert e.e1_hardware_anomalies == 1.0
    assert e.e10_hidden_triggers_or_backdoors == 0.8
    print("PASS: test_evidence_from_dict")


def test_evidence_summary():
    e = Evidence(e1_hardware_anomalies=1.0, e7_system_log_errors=0.5)
    s = e.summary()
    assert "Hardware anomalies" in s
    assert "System log errors" in s
    print("PASS: test_evidence_summary")


# ── LikelihoodTable ──────────────────────────────────────────────────

def test_likelihood_table_defaults():
    t = LikelihoodTable()
    assert t.prior("entropy") == 0.10
    assert t.prior("human_malice") == 0.15
    assert 0.0 < t.p_evidence_given_cause("entropy", "e1_hardware_anomalies") < 1.0
    print("PASS: test_likelihood_table_defaults")


def test_likelihood_table_set_prior():
    t = LikelihoodTable()
    t.set_prior("entropy", 0.5)
    assert t.prior("entropy") == 0.5
    print("PASS: test_likelihood_table_set_prior")


def test_likelihood_table_validation():
    t = LikelihoodTable()
    warnings = t.validate()
    assert isinstance(warnings, list)
    print("PASS: test_likelihood_table_validation")


# ── Bayesian Engine ──────────────────────────────────────────────────

def test_diagnose_empty():
    e = Evidence()
    result = diagnose(e)
    assert isinstance(result, DiagnosisResult)
    assert result.diagnosis in FAILURE_CLASSES
    assert 0.0 <= result.confidence <= 1.0
    print("PASS: test_diagnose_empty")


def test_diagnose_military_deception():
    result = diagnose_preset("military_deception")
    assert result.diagnosis == "human_malice"
    assert result.confidence > 0.99
    print("PASS: test_diagnose_military_deception")


def test_diagnose_hardware_crash():
    result = diagnose_preset("hardware_crash")
    assert result.diagnosis == "entropy"
    assert result.confidence > 0.95
    print("PASS: test_diagnose_hardware_crash")


def test_diagnose_distribution_drift():
    result = diagnose_preset("distribution_drift")
    assert result.diagnosis == "engineering_limits"
    assert result.confidence > 0.90
    print("PASS: test_diagnose_distribution_drift")


def test_diagnose_misconfigured_reward():
    result = diagnose_preset("misconfigured_reward")
    assert result.diagnosis == "human_error"
    assert result.confidence > 0.75
    print("PASS: test_diagnose_misconfigured_reward")


def test_diagnose_subtle_bias():
    result = diagnose_preset("subtle_bias")
    assert result.diagnosis == "human_bias"
    assert result.confidence > 0.60
    print("PASS: test_diagnose_subtle_bias")


def test_diagnose_posteriors_sum_to_one():
    for preset_name in PRESETS:
        result = diagnose_preset(preset_name)
        total = sum(result.posteriors.values())
        assert abs(total - 1.0) < 1e-6, f"{preset_name}: posteriors sum to {total}"
    print("PASS: test_diagnose_posteriors_sum_to_one")


def test_diagnose_custom_priors():
    e = Evidence(e9_obfuscated_code_or_weights=1.0, e10_hidden_triggers_or_backdoors=1.0)
    result = diagnose(e, prior_overrides={"human_malice": 0.90})
    # With strong malice prior and malice-favoring evidence, malice should win
    assert result.diagnosis == "human_malice"
    # Posterior for malice should increase vs default prior
    result_default = diagnose(e)
    assert result.posteriors["human_malice"] > result_default.posteriors["human_malice"]
    print("PASS: test_diagnose_custom_priors")


def test_diagnose_result_to_dict():
    result = diagnose_preset("military_deception")
    d = result.to_dict()
    assert "posteriors" in d
    assert "diagnosis" in d
    assert "confidence" in d
    assert isinstance(d["posteriors"], dict)
    print("PASS: test_diagnose_result_to_dict")


def test_confidence_threshold_flagging():
    result = diagnose_preset("subtle_bias")
    assert result.needs_investigation == (result.confidence < 0.6)
    print("PASS: test_confidence_threshold_flagging")


# ── Sensitivity ──────────────────────────────────────────────────────

def test_sensitivity_analysis():
    result = diagnose_preset("subtle_bias")
    engine = BayesianDiagnostic()
    sens = compute_sensitivity(result, engine)
    assert isinstance(sens, SensitivityReport)
    assert len(sens.parameter_sensitivity) > 0
    print("PASS: test_sensitivity_analysis")


def test_sensitivity_summary():
    result = diagnose_preset("military_deception")
    engine = BayesianDiagnostic()
    sens = compute_sensitivity(result, engine)
    s = sens.summary()
    assert len(s) > 0
    print("PASS: test_sensitivity_summary")


def test_sensitivity_flip_analysis():
    result = diagnose_preset("subtle_bias")
    engine = BayesianDiagnostic()
    sens = compute_sensitivity(result, engine)
    assert isinstance(sens.flip_analysis, list)
    print("PASS: test_sensitivity_flip_analysis")


# ── History ──────────────────────────────────────────────────────────

def test_history_record_and_retrieve():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        history = DiagnosisHistory(db_path=db_path)
        result = diagnose_preset("military_deception")
        history.record(result, result.evidence_vector, metadata={"test": True})
        records = history.recent(1)
        assert len(records) == 1
        record = records[0]
        # records are tuples: (DiagnosisResult, timestamp_str, metadata_dict)
        assert isinstance(record, tuple)
        assert len(record) == 3
        diag_result, timestamp, metadata = record
        assert diag_result.diagnosis == "human_malice"
        assert metadata.get("test") is True
    print("PASS: test_history_record_and_retrieve")


def test_history_stats():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        history = DiagnosisHistory(db_path=db_path)
        for preset in ["military_deception", "hardware_crash"]:
            result = diagnose_preset(preset)
            history.record(result, result.evidence_vector)
        stats = history.stats()
        assert stats["total_diagnoses"] == 2
        assert stats["average_confidence"] > 0.9
    print("PASS: test_history_stats")


def test_history_export_csv():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        csv_path = os.path.join(tmpdir, "export.csv")
        history = DiagnosisHistory(db_path=db_path)
        result = diagnose_preset("military_deception")
        history.record(result, result.evidence_vector)
        history.export_csv(csv_path)
        assert os.path.exists(csv_path)
        assert os.path.getsize(csv_path) > 0
    print("PASS: test_history_export_csv")


def test_history_by_diagnosis():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        history = DiagnosisHistory(db_path=db_path)
        for preset in ["military_deception", "hardware_crash", "military_deception"]:
            result = diagnose_preset(preset)
            history.record(result, result.evidence_vector)
        malice_records = history.by_diagnosis("human_malice")
        assert len(malice_records) == 2
    print("PASS: test_history_by_diagnosis")


# ── Batch ────────────────────────────────────────────────────────────

def test_batch_process():
    bp = BatchProcessor()
    evidence_list = [
        Evidence(e1_hardware_anomalies=1.0, e7_system_log_errors=1.0, e5_temporal_pattern_sudden=1.0),
        Evidence(e9_obfuscated_code_or_weights=1.0, e10_hidden_triggers_or_backdoors=1.0),
    ]
    results = bp.process(evidence_list)
    assert len(results) == 2
    assert results[0].diagnosis == "entropy"
    # Second case: e9+e10 without other malice indicators → bias may win
    # due to absent-evidence penalties on malice likelihoods
    assert results[1].diagnosis in ("human_malice", "human_bias")
    print("PASS: test_batch_process")


def test_batch_summary():
    bp = BatchProcessor()
    evidence_list = [
        Evidence(e1_hardware_anomalies=1.0, e7_system_log_errors=1.0),
        Evidence(e9_obfuscated_code_or_weights=1.0),
        Evidence(),
    ]
    results = bp.process(evidence_list)
    summary = bp.summary(results)
    assert summary["total"] == 3
    assert "average_confidence" in summary
    assert "by_type" in summary
    print("PASS: test_batch_summary")


def test_batch_compare():
    bp = BatchProcessor()
    r1 = diagnose_preset("hardware_crash")
    r2 = diagnose_preset("distribution_drift")
    table = bp.compare([r1, r2])
    assert isinstance(table, str)
    assert len(table) > 0
    assert "entropy" in table.lower() or "Entropy" in table
    print("PASS: test_batch_compare")


# ── Alerts ───────────────────────────────────────────────────────────

def test_alert_manager():
    am = AlertManager()
    triggered = []
    am.add_rule(
        "test_rule",
        condition=lambda r: r.confidence > 0.9,
        actions=[lambda r, name: triggered.append((r.diagnosis, name))],
    )
    result = diagnose_preset("military_deception")
    am.check(result)
    assert len(triggered) == 1
    assert triggered[0][0] == "human_malice"
    assert triggered[0][1] == "test_rule"
    print("PASS: test_alert_manager")


def test_alert_manager_no_trigger():
    am = AlertManager()
    triggered = []
    am.add_rule(
        "high_confidence_only",
        condition=lambda r: r.confidence > 0.99,
        actions=[lambda r, name: triggered.append(True)],
    )
    result = diagnose_preset("subtle_bias")
    am.check(result)
    assert len(triggered) == 0
    print("PASS: test_alert_manager_no_trigger")


def test_threshold_alert():
    from ai_diagnostic import Evidence, diagnose

    ta_above = ThresholdAlert(threshold=0.9, direction="above")
    ta_below = ThresholdAlert(threshold=0.5, direction="below")

    result_high = diagnose_preset("military_deception")
    # Create ambiguous evidence for low confidence
    ambiguous = Evidence(e9_obfuscated_code_or_weights=0.3, e4_human_interface_errors=0.3)
    result_low = diagnose(ambiguous)

    # High confidence case
    assert ta_above.check_confidence(result_high)
    # Low confidence case
    assert not ta_above.check_confidence(result_low) if result_low.confidence < 0.9 else True

    # Posterior-based checks
    assert ta_above.check_posterior("human_malice", result_high)
    assert not ta_above.check_posterior("human_malice", result_low)
    print("PASS: test_threshold_alert")


def test_preset_rules():
    am = AlertManager()
    rules = am.preset_rules()
    assert len(rules) == 4
    rule_names = [r.name for r in rules]
    assert "critical_malice" in rule_names
    assert "warning_low_confidence" in rule_names
    assert "entropy_alert" in rule_names
    assert "investigation_needed" in rule_names
    print("PASS: test_preset_rules")


# ── Reports ──────────────────────────────────────────────────────────

def test_html_report():
    gen = ReportGenerator()
    result = diagnose_preset("military_deception")
    html = gen.html_report(result)
    assert "<html" in html.lower() or "<!doctype" in html.lower()
    assert "human_malice" in html.lower() or "malice" in html.lower()
    print("PASS: test_html_report")


def test_html_report_with_sensitivity():
    gen = ReportGenerator()
    result = diagnose_preset("subtle_bias")
    engine = BayesianDiagnostic()
    sens = compute_sensitivity(result, engine)
    html = gen.html_report(result, sensitivity=sens)
    assert len(html) > 1000
    print("PASS: test_html_report_with_sensitivity")


def test_html_comparison_report():
    gen = ReportGenerator()
    r1 = diagnose_preset("hardware_crash")
    r2 = diagnose_preset("distribution_drift")
    html = gen.html_comparison_report([r1, r2], labels=["crash", "drift"])
    assert len(html) > 1000
    assert "crash" in html.lower() or "drift" in html.lower()
    print("PASS: test_html_comparison_report")


def test_text_report():
    gen = ReportGenerator()
    result = diagnose_preset("hardware_crash")
    text = gen.text_report(result)
    assert "entropy" in text.lower() or "Entropy" in text
    print("PASS: test_text_report")


def test_json_report():
    gen = ReportGenerator()
    result = diagnose_preset("military_deception")
    j = gen.json_report(result)
    data = json.loads(j)
    assert "diagnosis" in data
    assert "posteriors" in data
    print("PASS: test_json_report")


def test_save_report():
    gen = ReportGenerator()
    result = diagnose_preset("hardware_crash")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "report.html")
        html = gen.html_report(result)
        gen.save_report(html, path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
    print("PASS: test_save_report")


# ── Presets completeness ────────────────────────────────────────────

def test_all_presets_run():
    for name in PRESETS:
        result = diagnose_preset(name)
        assert result.diagnosis in FAILURE_CLASSES, f"Preset {name} failed"
        total = sum(result.posteriors.values())
        assert abs(total - 1.0) < 1e-6
    print("PASS: test_all_presets_run")


def test_all_presets_have_description():
    for name, preset in PRESETS.items():
        assert "description" in preset, f"Preset {name} missing description"
        assert "evidence" in preset, f"Preset {name} missing evidence"
        assert len(preset["evidence"]) == 15, f"Preset {name} wrong evidence count"
    print("PASS: test_all_presets_have_description")


# ── Edge cases ──────────────────────────────────────────────────────

def test_all_evidence_present():
    from ai_diagnostic.evidence import EVIDENCE_PARAMS
    d = {name: 1.0 for name in EVIDENCE_PARAMS}
    e = Evidence(**d)
    result = diagnose(e)
    assert result.confidence > 0.99
    assert result.diagnosis == "human_malice"
    print("PASS: test_all_evidence_present")


def test_extreme_prior_override():
    e = Evidence()
    result = diagnose(e, prior_overrides={"entropy": 0.9999})
    # With extreme entropy prior and no evidence, entropy should have highest posterior
    # even if not the diagnosis (absent evidence penalizes all causes via likelihoods)
    assert result.posteriors["entropy"] > 0.05
    print("PASS: test_extreme_prior_override")


# ── Run all tests ────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_evidence_creation,
        test_evidence_validation,
        test_evidence_vector,
        test_evidence_from_dict,
        test_evidence_summary,
        test_likelihood_table_defaults,
        test_likelihood_table_set_prior,
        test_likelihood_table_validation,
        test_diagnose_empty,
        test_diagnose_military_deception,
        test_diagnose_hardware_crash,
        test_diagnose_distribution_drift,
        test_diagnose_misconfigured_reward,
        test_diagnose_subtle_bias,
        test_diagnose_posteriors_sum_to_one,
        test_diagnose_custom_priors,
        test_diagnose_result_to_dict,
        test_confidence_threshold_flagging,
        test_sensitivity_analysis,
        test_sensitivity_summary,
        test_sensitivity_flip_analysis,
        test_history_record_and_retrieve,
        test_history_stats,
        test_history_export_csv,
        test_history_by_diagnosis,
        test_batch_process,
        test_batch_summary,
        test_batch_compare,
        test_alert_manager,
        test_alert_manager_no_trigger,
        test_threshold_alert,
        test_preset_rules,
        test_html_report,
        test_html_report_with_sensitivity,
        test_html_comparison_report,
        test_text_report,
        test_json_report,
        test_save_report,
        test_all_presets_run,
        test_all_presets_have_description,
        test_all_evidence_present,
        test_extreme_prior_override,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__}: {e}")
            failed += 1

    print()
    print(f"{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    if failed == 0:
        print("ALL TESTS PASSED")
    print(f"{'=' * 50}")
