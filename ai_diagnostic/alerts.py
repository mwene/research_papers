"""
Alert and notification system for diagnostic results.

Provides rule-based alerting with configurable actions including
webhooks, email, logging, and console output.
"""

import json
import smtplib
import sys
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.message import EmailMessage
from typing import Callable, List, Optional

from .bayesian import DiagnosisResult
from .likelihoods import FAILURE_CLASSES


@dataclass
class AlertRule:
    name: str
    condition: Callable[[DiagnosisResult], bool]
    actions: List[Callable[[DiagnosisResult, str], None]]


class ThresholdAlert:
    """Helper for confidence/posterior threshold checks."""

    def __init__(self, threshold: float, direction: str = "above"):
        if direction not in ("above", "below"):
            raise ValueError(f"direction must be 'above' or 'below', got {direction!r}")
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        self.threshold = threshold
        self.direction = direction

    def check_confidence(self, result: DiagnosisResult) -> bool:
        if self.direction == "above":
            return result.confidence > self.threshold
        return result.confidence < self.threshold

    def check_posterior(self, cause: str, result: DiagnosisResult) -> bool:
        if cause not in FAILURE_CLASSES:
            raise ValueError(f"Unknown cause: {cause}")
        prob = result.posteriors.get(cause, 0.0)
        if self.direction == "above":
            return prob > self.threshold
        return prob < self.threshold


class AlertManager:
    """Rule-based alert manager for diagnosis results."""

    def __init__(self):
        self._rules: List[AlertRule] = []
        self._triggered: List[str] = []

    def add_rule(
        self,
        name: str,
        condition: Callable[[DiagnosisResult], bool],
        actions: List[Callable[[DiagnosisResult, str], None]],
    ) -> None:
        self._rules.append(AlertRule(name=name, condition=condition, actions=actions))

    def check(self, result: DiagnosisResult) -> List[str]:
        self._triggered.clear()
        for rule in self._rules:
            if rule.condition(result):
                self._triggered.append(rule.name)
                for action in rule.actions:
                    action(result, rule.name)
        return list(self._triggered)

    def add_webhook(
        self, name: str, url: str, method: str = "POST"
    ) -> Callable[[DiagnosisResult, str], None]:
        def webhook_action(result: DiagnosisResult, rule_name: str) -> None:
            payload = json.dumps({
                "rule": rule_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **result.to_dict(),
            }).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method=method,
            )
            try:
                urllib.request.urlopen(req, timeout=10)
            except Exception as exc:
                print(
                    f"[AlertManager] webhook {name!r} failed: {exc}",
                    file=sys.stderr,
                )

        return webhook_action

    @staticmethod
    def add_log_action(path: str) -> Callable[[DiagnosisResult, str], None]:
        def log_action(result: DiagnosisResult, rule_name: str) -> None:
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "rule": rule_name,
                **result.to_dict(),
            }
            with open(path, "a") as f:
                f.write(json.dumps(entry) + "\n")

        return log_action

    @staticmethod
    def add_console_action() -> Callable[[DiagnosisResult, str], None]:
        def console_action(result: DiagnosisResult, rule_name: str) -> None:
            label = FAILURE_CLASSES[
                list(FAILURE_CLASSES).index(result.diagnosis)
            ] if result.diagnosis in FAILURE_CLASSES else result.diagnosis
            print(
                f"[ALERT] {rule_name}: diagnosis={result.diagnosis} "
                f"confidence={result.confidence:.2%}",
                file=sys.stderr,
            )

        return console_action

    @staticmethod
    def add_email_action(
        smtp_host: str, to: str, from_addr: str
    ) -> Callable[[DiagnosisResult, str], None]:
        def email_action(result: DiagnosisResult, rule_name: str) -> None:
            msg = EmailMessage()
            msg["Subject"] = f"[AI Diagnostic Alert] {rule_name}"
            msg["From"] = from_addr
            msg["To"] = to
            body = (
                f"Alert triggered: {rule_name}\n"
                f"Diagnosis: {result.diagnosis}\n"
                f"Confidence: {result.confidence:.2%}\n"
                f"Needs investigation: {result.needs_investigation}\n\n"
                f"Posterior probabilities:\n"
            )
            for cause, prob in sorted(
                result.posteriors.items(), key=lambda x: x[1], reverse=True
            ):
                body += f"  {cause}: {prob:.4f}\n"
            msg.set_content(body)
            try:
                with smtplib.SMTP(smtp_host, timeout=10) as smtp:
                    smtp.send_message(msg)
            except Exception as exc:
                print(
                    f"[AlertManager] email to {to} failed: {exc}",
                    file=sys.stderr,
                )

        return email_action

    def preset_rules(self) -> List[AlertRule]:
        presets = [
            AlertRule(
                name="critical_malice",
                condition=lambda r: r.diagnosis == "human_malice" and r.confidence > 0.9,
                actions=[self.add_console_action()],
            ),
            AlertRule(
                name="warning_low_confidence",
                condition=lambda r: r.confidence < 0.5,
                actions=[self.add_console_action()],
            ),
            AlertRule(
                name="entropy_alert",
                condition=lambda r: r.diagnosis == "entropy" and r.confidence > 0.8,
                actions=[self.add_console_action()],
            ),
            AlertRule(
                name="investigation_needed",
                condition=lambda r: r.needs_investigation,
                actions=[self.add_console_action()],
            ),
        ]
        self._rules.extend(presets)
        return presets
