use crate::params::Resolved;
use crate::quantities::{hitting_prob, time_to_reach, urgency};
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct FiveQuantities {
    pub expected_net_gain: f64,
    pub uncertainty_band: f64,
    pub time_to_goal_months: Option<f64>,
    pub floor_risk_if_act: f64,
    pub floor_risk_if_wait: f64,
    pub urgency: f64,
}

pub(crate) fn compute_five(p: &Resolved) -> FiveQuantities {
    let net_rate = p.gain_rate - p.wait_drift;
    let expected_net_gain = net_rate * p.horizon;
    let uncertainty_band = p.volatility * p.horizon.sqrt();

    let floor_level = (p.state_now - p.floor_distance).max(0.0);
    let risk_act = hitting_prob(p.state_now, p.gain_rate, p.volatility, p.horizon, floor_level);
    let risk_wait = hitting_prob(p.state_now, p.wait_drift, p.volatility, p.horizon, floor_level);

    FiveQuantities {
        expected_net_gain: expected_net_gain,
        uncertainty_band,
        time_to_goal_months: time_to_reach(1.0 - p.state_now, p.gain_rate.max(0.0)),
        floor_risk_if_act: risk_act,
        floor_risk_if_wait: risk_wait,
        urgency: urgency(p.forced_in, p.horizon),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::DecisionParams;

    fn base() -> Resolved {
        DecisionParams::default().resolved()
    }

    #[test]
    fn waiting_decay_raises_floor_risk_vs_acting() {
        let mut p = base();
        p.gain_rate = 0.03;
        p.wait_drift = -0.03;
        let five = compute_five(&p);
        assert!(five.floor_risk_if_wait > five.floor_risk_if_act);
    }

    #[test]
    fn band_grows_with_volatility_and_time() {
        let mut p = base();
        p.volatility = 0.1;
        p.horizon = 9.0;
        let b1 = compute_five(&p).uncertainty_band;
        p.volatility = 0.3;
        let b2 = compute_five(&p).uncertainty_band;
        p.volatility = 0.1;
        p.horizon = 36.0;
        let b3 = compute_five(&p).uncertainty_band;
        assert!(b2 > b1 && b3 > b1);
    }

    #[test]
    fn unreachable_goal_is_none() {
        let mut p = base();
        p.gain_rate = -0.01;
        assert!(compute_five(&p).time_to_goal_months.is_none());
    }

    #[test]
    fn net_gain_uses_relative_rates() {
        let mut p = base();
        p.gain_rate = 0.05;
        p.wait_drift = 0.01;
        p.horizon = 10.0;
        let g = compute_five(&p).expected_net_gain;
        assert!((g - 0.4).abs() < 1e-9);
    }
}
