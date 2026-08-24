use crate::advice::{advice_lines, headline};
use crate::five::{compute_five, FiveQuantities};
use crate::params::Resolved;
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Action {
    ActNow,
    Wait,
    Drop,
}

#[derive(Debug, Clone, Serialize)]
pub struct Scores {
    pub act_now: f64,
    pub wait: f64,
    pub drop: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Verdict {
    pub action: Action,
    pub headline: String,
    pub scores: Scores,
    pub decision_value: f64,
    pub confidence: f64,
    pub five: FiveQuantities,
    pub advice: Vec<String>,
}

fn softmax3(a: f64, b: f64, c: f64, tau: f64) -> (f64, f64, f64) {
    let m = a.max(b).max(c);
    let ea = ((a - m) / tau).exp();
    let eb = ((b - m) / tau).exp();
    let ec = ((c - m) / tau).exp();
    let s = ea + eb + ec;
    (ea / s, eb / s, ec / s)
}

pub fn decide(p: &Resolved) -> Verdict {
    let q = compute_five(p);
    let theta1 = q.expected_net_gain;
    let urg = q.urgency;

    let survival = (-2.2 * urg).exp();
    let t_eff = p
        .forced_in
        .map(|tf| p.horizon.min(tf))
        .unwrap_or(p.horizon);
    let info_value =
        0.9 * p.volatility * t_eff.sqrt() * (0.3 + theta1.max(0.0)) * (1.0 - 0.6 * urg);
    let forced_penalty = urg * urg * (0.45 + 1.3 * p.floor_severity);

    let u_act = theta1 - 0.85 * p.act_cost - 2.2 * p.floor_severity * q.floor_risk_if_act;
    let u_wait = theta1 * survival + info_value - forced_penalty
        - 2.2 * p.floor_severity * q.floor_risk_if_wait;
    let u_drop =
        -0.95 * theta1.max(0.0) - 1.6 * p.floor_severity * q.floor_risk_if_wait
            + 0.6 * (-theta1.min(0.0));

    let (pa, pw, pd) = softmax3(u_act, u_wait, u_drop, 0.30);

    let (action, top, second) = if pa >= pw && pa >= pd {
        (Action::ActNow, pa, pw.max(pd))
    } else if pw >= pd {
        (Action::Wait, pw, pa.max(pd))
    } else {
        (Action::Drop, pd, pa.max(pw))
    };

    let confidence = ((0.35 + 0.65 * (top - second)) * (0.55 + 0.45 * p.data_quality)).clamp(0.0, 1.0);

    let advice = advice_lines(action, p, &q);

    Verdict {
        action,
        headline: headline(action, p, &q),
        scores: Scores { act_now: pa, wait: pw, drop: pd },
        decision_value: top,
        confidence,
        five: q,
        advice,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::DecisionParams;

    fn from(params: DecisionParams) -> Verdict {
        decide(&params.resolved())
    }

    #[test]
    fn scores_form_probability_simplex() {
        let v = from(DecisionParams::default());
        let s = v.scores.act_now + v.scores.wait + v.scores.drop;
        assert!((s - 1.0).abs() < 1e-9);
        assert!((0.0..=1.0).contains(&v.decision_value));
        assert!((0.0..=1.0).contains(&v.confidence));
    }

    #[test]
    fn strong_deal_acts() {
        let mut d = DecisionParams::default();
        d.state_now = Some(0.45);
        d.floor_distance = Some(0.4);
        d.gain_rate = Some(0.06);
        d.wait_drift = Some(-0.01);
        d.volatility = Some(0.10);
        d.horizon_months = Some(12.0);
        d.forced_in_months = Some(24.0);
        d.act_cost = Some(0.2);
        d.floor_severity = Some(0.1);
        assert_eq!(from(d).action, Action::ActNow);
    }

    #[test]
    fn uncertain_open_window_waits() {
        let mut d = DecisionParams::default();
        d.state_now = Some(0.55);
        d.floor_distance = Some(0.5);
        d.gain_rate = Some(0.02);
        d.wait_drift = Some(0.0);
        d.volatility = Some(0.5);
        d.horizon_months = Some(12.0);
        d.forced_in_months = Some(900.0);
        d.act_cost = Some(0.5);
        d.floor_severity = Some(0.05);
        assert_eq!(from(d).action, Action::Wait);
    }

    #[test]
    fn losing_prospect_drops() {
        let mut d = DecisionParams::default();
        d.state_now = Some(0.5);
        d.floor_distance = Some(0.6);
        d.gain_rate = Some(-0.02);
        d.wait_drift = Some(0.0);
        d.volatility = Some(0.08);
        d.horizon_months = Some(12.0);
        d.forced_in_months = Some(900.0);
        d.act_cost = Some(0.4);
        d.floor_severity = Some(0.05);
        assert_eq!(from(d).action, Action::Drop);
    }

    #[test]
    fn shrinking_window_flips_wait_to_act() {
        let mut d = DecisionParams::default();
        d.gain_rate = Some(0.04);
        d.wait_drift = Some(-0.005);
        d.volatility = Some(0.25);
        d.horizon_months = Some(12.0);
        d.forced_in_months = Some(60.0);
        d.act_cost = Some(0.3);
        d.floor_severity = Some(0.2);
        let far = from(d.clone()).action;
        assert_eq!(far, Action::Wait, "far window should favor waiting");

        let mut d2 = d.clone();
        d2.forced_in_months = Some(1.0);
        let near = from(d2).action;
        assert_eq!(near, Action::ActNow, "tight window must force early action");
    }

    #[test]
    fn severe_floor_nearby_protects_by_acting() {
        let mut d = DecisionParams::default();
        d.state_now = Some(0.30);
        d.floor_distance = Some(0.12);
        d.gain_rate = Some(0.03);
        d.wait_drift = Some(-0.06);
        d.volatility = Some(0.20);
        d.horizon_months = Some(12.0);
        d.forced_in_months = Some(18.0);
        d.act_cost = Some(0.4);
        d.floor_severity = Some(0.9);
        assert_eq!(from(d).action, Action::ActNow);
    }

    #[test]
    fn advice_and_headline_present() {
        let v = from(DecisionParams::default());
        assert!(!v.headline.is_empty());
        assert!(v.advice.len() >= 2);
    }
}
